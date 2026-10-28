#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#include "macros/cpp_defines.h"
#include "bench_common.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
    #include "macros/macrolib.h"
    #include "time_it.h"
    #include "parallel_util.h"
    #include "array_metrics.h"

    #if DOUBLE == 0
        #define VTI   i32
        #define VTF   f32
        #define VTM   m32
        #define VEC_SCALE_SHIFT  2
        #define VEC_LEN  vec_len_default_f32
    #elif DOUBLE == 1
        #define VTI   i64
        #define VTF   f64
        #define VTM   m64
        #define VEC_SCALE_SHIFT  3
        #define VEC_LEN  vec_len_default_f64
    #endif

    #include "vectorization/vectorization_gen.h"
#ifdef __cplusplus
}
#endif

INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;

double * thread_time_compute, * thread_time_barrier;

// ==========================================================================================================================================
// = Hilbert Sorting Utilities
// ==========================================================================================================================================

// Converts 2D coordinates (row, col) into a 1D Hilbert Curve distance
static inline uint64_t hilbert_c2d(uint32_t x, uint32_t y) {
    uint64_t d = 0;
    // Process 32 bits from MSB to LSB
    for (int s = 31; s >= 0; s--) {
        uint32_t rx = (x >> s) & 1;
        uint32_t ry = (y >> s) & 1;
        
        // Append quadrant bits to distance
        d = (d << 2) | ((3 * rx) ^ ry);
        
        // Rotate and flip coordinates for the next iteration if necessary
        if (ry == 0) {
            if (rx == 1) {
                x = ~x;
                y = ~y;
            }
            // Swap x and y
            uint32_t t = x;
            x = y;
            y = t;
        }
    }
    return d;
}

struct SortContext {
    uint64_t * hilbert_pos;
};

int
hilbert_comp_context(const void * a_ptr, const void * b_ptr, void * arg)
{
    long a_idx = *((long *) a_ptr);
    long b_idx = *((long *) b_ptr);
    struct SortContext * ctx = (struct SortContext *) arg;
    
    if (ctx->hilbert_pos[a_idx] > ctx->hilbert_pos[b_idx]) return 1;
    if (ctx->hilbert_pos[a_idx] < ctx->hilbert_pos[b_idx]) return -1;
    return 0;
}

// ==========================================================================================================================================
// = Data Structure
// ==========================================================================================================================================

struct COOHilbertSegmentedArrays : Matrix_Format
{
    // Reordered Hilbert Arrays 
    INT_T * row_ind_h; 
    INT_T * col_ind_h; 
    ValueType * a_h;   

    // Tiny arrays to cache row boundaries for O(1) lookup
    INT_T * thread_min_r;
    INT_T * thread_max_r;

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    long num_loops;

    COOHilbertSegmentedArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k) 
        : Matrix_Format(m, n, nnz, k)
    {
        int num_threads = omp_get_max_threads();
        double time_balance;
        
        // Temporary Original Arrays
        INT_T * row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
        INT_T * col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
        ValueType * a = (ValueType *) malloc(nnz * sizeof(*a));

        row_ind_h = (INT_T *) malloc(nnz * sizeof(*row_ind_h));
        col_ind_h = (INT_T *) malloc(nnz * sizeof(*col_ind_h));
        a_h = (ValueType *) malloc(nnz * sizeof(*a_h));
        
        // 1. Initial CSR -> COO Copy 
        #pragma omp parallel for schedule(dynamic, 1024)
        for (long i = 0; i < m; i++) {
            for (long j = csr_ia[i]; j < csr_ia[i+1]; j++) {
                row_ind[j] = i;
                col_ind[j] = csr_ja[j];
                a[j] = csr_a[j];
            }
        }

        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
        
        // 2. Partitioning (Snapped to Row Boundaries)
        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                long raw_s, raw_e;
                
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &raw_s, &raw_e);
                
                if (tnum > 0 && raw_s < nnz) {
                    while(raw_s < nnz && row_ind[raw_s] == row_ind[raw_s - 1]) {
                        raw_s++;
                    }
                }
                thread_j_s[tnum] = raw_s;
                
                _Pragma("omp barrier")
                
                if (tnum == num_threads - 1) {
                    thread_j_e[tnum] = nnz;
                } else {
                    thread_j_e[tnum] = thread_j_s[tnum + 1];
                }

                if (thread_j_e[tnum] < thread_j_s[tnum]) {
                    thread_j_e[tnum] = thread_j_s[tnum];
                }
            }
        );

        // 3. Segmented Hilbert Sorting
        uint64_t * hilbert_pos = (uint64_t *) malloc(nnz * sizeof(uint64_t));
        
        #pragma omp parallel for schedule(static)
        for(long i=0; i<nnz; i++) {
             hilbert_pos[i] = hilbert_c2d((uint32_t)row_ind[i], (uint32_t)col_ind[i]);
        }

        thread_min_r = (INT_T *) malloc(num_threads * sizeof(*thread_min_r));
        thread_max_r = (INT_T *) malloc(num_threads * sizeof(*thread_max_r));

        #pragma omp parallel
        {
            int tnum = omp_get_thread_num();
            long js = thread_j_s[tnum];
            long je = thread_j_e[tnum];
            long size = je - js;
            
            // Cache boundaries for O(1) lookup during compute
            if (je > js) {
                thread_min_r[tnum] = row_ind[js];
                thread_max_r[tnum] = row_ind[je - 1];
            } else {
                thread_min_r[tnum] = -1;
                thread_max_r[tnum] = -1;
            }

            if (size > 0) {
                long * perm = (long *) malloc(size * sizeof(long));
                for(long i=0; i<size; i++) {
                    perm[i] = js + i; 
                }
                
                struct SortContext ctx = { hilbert_pos };
                qsort_r(perm, size, sizeof(long), hilbert_comp_context, &ctx);
                
                // Apply permutation into Hilbert arrays
                for(long i=0; i<size; i++) {
                    long src_idx = perm[i];
                    long dst_idx = js + i;
                    
                    row_ind_h[dst_idx] = row_ind[src_idx];
                    col_ind_h[dst_idx] = col_ind[src_idx];
                    a_h[dst_idx] = a[src_idx];
                }
                
                free(perm);
            }
        }
        
        free(hilbert_pos);
        // Free the unsorted original arrays to cut memory footprint in half
        free(row_ind);
        free(col_ind);
        free(a);

        #ifdef PRINT_STATISTICS
            long i;
            num_loops = 0;
            thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
            thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
            for (i=0;i<num_threads;i++)
            {
                printf("Thread %ld: nnz range [%d, %d) nnz: %ld of nnz_total: %ld\n", i, thread_j_s[i], thread_j_e[i], thread_j_e[i] - thread_j_s[i], nnz);
            }
        #endif
    }

    ~COOHilbertSegmentedArrays()
    {
        free(a_h); free(row_ind_h); free(col_ind_h);
        free(thread_j_s); free(thread_j_e);
        free(thread_min_r); free(thread_max_r);
        #ifdef PRINT_STATISTICS
            free(thread_time_barrier); free(thread_time_compute);
        #endif
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

// ==========================================================================================================================================
// = Forward declarations
// ==========================================================================================================================================

void compute_cooh_vector_xrow(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_cooh_sddmm(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

// ==========================================================================================================================================
// = Interface Implementations
// ==========================================================================================================================================

void
COOHilbertSegmentedArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;
    compute_cooh_vector_xrow(this, x, y, k);
}

void
COOHilbertSegmentedArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    compute_cooh_sddmm(this, x, y, out, k);
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COOHilbertSegmentedArrays * coo = new COOHilbertSegmentedArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T)); 
    coo->format_name = (char *) "COO_Hilbert";
    return coo;
}

// ==========================================================================================================================================
// = Subkernels COO Hilbert
// ==========================================================================================================================================

__attribute__((hot))
static inline
void
subkernel_val_cooh_vec_noatomic(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
{
    // Read from the Hilbert-ordered arrays
    long r = coo->row_ind_h[j];
    long c_idx = coo->col_ind_h[j];
    ValueType val = coo->a_h[j];

    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);
    
    vec_t(VTF, VEC_LEN) v_val, v_x, v_prod, v_y;
    
    c_e_vector = k & mask;
    v_val = vec_set1(VTF, VEC_LEN, val);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_y = vec_loadu(VTF, VEC_LEN, &y[r * k + c]);
        v_x = vec_loadu(VTF, VEC_LEN, &x[c_idx * k + c]);
        v_prod = vec_fmadd(VTF, VEC_LEN, v_val, v_x, v_y); 
        vec_storeu(VTF, VEC_LEN, &y[r * k + c], v_prod);
    }

    for (c = c_e_vector; c < k; c++) {
        y[r * k + c] += val * x[c_idx * k + c];
    }
}

__attribute__((hot))
static inline
void
subkernel_val_cooh_sddmm(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, long j, int k)
{
    long r = coo->row_ind_h[j];
    long c_idx = coo->col_ind_h[j];
    ValueType val = coo->a_h[j];

    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);
    
    vec_t(VTF, VEC_LEN) v_x, v_y, v_sum;
    c_e_vector = k & mask;

    v_sum = vec_set1(VTF, VEC_LEN, 0);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_x = vec_loadu(VTF, VEC_LEN, &x[r * k + c]);
        v_y = vec_loadu(VTF, VEC_LEN, &y[c_idx * k + c]);
        v_sum = vec_fmadd(VTF, VEC_LEN, v_x, v_y, v_sum);
    }
    
    ValueType dot_prod = vec_reduce_add(VTF, VEC_LEN, v_sum);

    for (c = c_e_vector; c < k; c++) {
        dot_prod += x[r * k + c] * y[c_idx * k + c];
    }

    out[j] = dot_prod * val;
}

// ==========================================================================================================================================
// = COO Main Computation Kernels
// ==========================================================================================================================================

void
compute_cooh_vector_xrow(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];
        
        // // 1. Safe Initialization using O(1) Cached Boundaries
        // long min_r = coo->thread_min_r[tnum];
        // long max_r = coo->thread_max_r[tnum];

        // if (min_r != -1 && max_r != -1) {
        //     for (long r = min_r; r <= max_r; r++) {
        //         for (long c = 0; c < k; c++) {
        //             y[r * k + c] = 0.0;
        //         }
        //     }
        // }

        long min_r = -1, max_r = -1;
        if (j_e > j_s) {
            min_r = coo->row_ind_h[j_s];
            max_r = coo->row_ind_h[j_s];
            for(long j = j_s; j < j_e; j++) {
                if (coo->row_ind_h[j] < min_r) min_r = coo->row_ind_h[j];
                if (coo->row_ind_h[j] > max_r) max_r = coo->row_ind_h[j];
            }
        }
        
        // Zero out the Y buffer for the rows we own
        if (min_r != -1) {
            long total_vals = (max_r - min_r + 1) * k;
            // We can't memset broadly if the rows aren't contiguous in ID, 
            // but Z-order blocks usually keep rows somewhat together.
            // However, to be 100% correct without assuming contiguous rows, 
            // we should iterate 0 to k for every unique row.
            // But since we own the whole range [min_r, max_r] implicitly by the partitioner logic (assuming standard CSR input), 
            // we can just wipe the whole range.
            for (long r = min_r; r <= max_r; r++) {
                for (long c = 0; c < k; c++) y[r * k + c] = 0.0;
            }
        }
        
        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif
        
        // 2. Compute using Hilbert-ordered arrays
        for (long j = j_s; j < j_e; j++)
        {
            subkernel_val_cooh_vec_noatomic(coo, x, y, j, k);
        }

        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1, _Pragma("omp barrier"));
        thread_time_barrier[tnum] += time;
        #endif
    }
}

void
compute_cooh_sddmm(COOHilbertSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k)
{
    if (coo->out == NULL) coo->out = out;
    if (coo->x == NULL) { coo->x = x; coo->y = y; }

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];

        for (long j = j_s; j < j_e; j++)
        {
            subkernel_val_cooh_sddmm(coo, x, y, out, j, k);
        }
    }
}

// ==========================================================================================================================================
// = Statistics
// ==========================================================================================================================================

void
COOHilbertSegmentedArrays::statistics_start()
{
    int num_threads = omp_get_max_threads();
    long i;
    num_loops = 0;
    for (i=0;i<num_threads;i++)
    {
        thread_time_compute[i] = 0;
        thread_time_barrier[i] = 0;
    }
}

int
COOHilbertSegmentedArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}