#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h> // For memset

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

int prefetch_distance = atoi(getenv("PREFETCH_DIST"));

// ============================================================
// Z-Order Utilities (Reused and Adapted)
// ============================================================

// Structure to bundle the sort Context
struct SortContext {
    uint64_t * z_pos;
};

int
z_order_comp_context(const void * a_ptr, const void * b_ptr, void * arg)
{
    long a_idx = *((long *) a_ptr); // Index into the global array
    long b_idx = *((long *) b_ptr);
    struct SortContext * ctx = (struct SortContext *) arg;
    
    // Sort by Z-value
    if (ctx->z_pos[a_idx] > ctx->z_pos[b_idx]) return 1;
    if (ctx->z_pos[a_idx] < ctx->z_pos[b_idx]) return -1;
    return 0;
}


// ============================================================
// COO Segmented Z-Order Structure
// ============================================================

INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;
double * thread_time_compute, * thread_time_barrier;

struct COOZSegmentedArrays : Matrix_Format
{
    INT_T * row_ind; 
    INT_T * col_ind; 
    ValueType * a;   

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    long num_loops;

    COOZSegmentedArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k)
    {
        int num_threads = omp_get_max_threads();
        
        row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
        col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
        a = (ValueType *) malloc(nnz * sizeof(*a));
        
        // 1. Initial CSR -> COO Copy (Unsorted)
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel
        {
            int tnum = omp_get_thread_num();
            long raw_s, raw_e;
            
            // Balanced split of NNZ
            loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &raw_s, &raw_e);
            
            // Snap start to next row if we are in the middle of one
            if (tnum > 0 && raw_s < nnz) {
                while(raw_s < nnz && row_ind[raw_s] == row_ind[raw_s - 1]) {
                    raw_s++;
                }
            }
            thread_j_s[tnum] = raw_s;
            #pragma omp barrier
            
            // Define end based on neighbor
            if (tnum == num_threads - 1) thread_j_e[tnum] = nnz;
            else thread_j_e[tnum] = thread_j_s[tnum + 1];

            // Safety clamp
            if (thread_j_e[tnum] < thread_j_s[tnum]) thread_j_e[tnum] = thread_j_s[tnum];
        }

        // 3. Segmented Z-Order Sorting
        // We sort the chunks locally. This preserves the row partitioning (no shared rows)
        // while optimizing locality *inside* the chunk.
        
        // Compute Z-values globally first
        uint64_t * z_pos = (uint64_t *) malloc(nnz * sizeof(uint64_t));
        #pragma omp parallel for schedule(static)
        for(long i=0; i<nnz; i++) {
             z_pos[i] = bits_u32_interleave((uint32_t)row_ind[i], (uint32_t)col_ind[i]);
        }

        // Parallel Sort of Segments
        #pragma omp parallel
        {
            int tnum = omp_get_thread_num();
            long js = thread_j_s[tnum];
            long je = thread_j_e[tnum];
            long size = je - js;
            
            if (size > 0) {
                // Create a permutation array for this segment
                long * perm = (long *) malloc(size * sizeof(long));
                for(long i=0; i<size; i++) perm[i] = js + i; // Global indices
                
                struct SortContext ctx = { z_pos };
                
                // Sort the permutation array based on Z-values
                qsort_r(perm, size, sizeof(long), z_order_comp_context, &ctx);
                
                // Apply permutation to local segment
                // (Need temporary buffers to swap data)
                INT_T * t_row = (INT_T *) malloc(size * sizeof(INT_T));
                INT_T * t_col = (INT_T *) malloc(size * sizeof(INT_T));
                ValueType * t_val = (ValueType *) malloc(size * sizeof(ValueType));
                
                for(long i=0; i<size; i++) {
                    long src_idx = perm[i];
                    t_row[i] = row_ind[src_idx];
                    t_col[i] = col_ind[src_idx];
                    t_val[i] = a[src_idx];
                }
                
                // Copy back
                for(long i=0; i<size; i++) {
                    long dst_idx = js + i;
                    row_ind[dst_idx] = t_row[i];
                    col_ind[dst_idx] = t_col[i];
                    a[dst_idx] = t_val[i];
                }
                
                free(perm); free(t_row); free(t_col); free(t_val);
            }
        }
        
        free(z_pos);

        #ifdef PRINT_STATISTICS
             // Alloc stats...
             thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
             thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
        #endif
    }

    ~COOZSegmentedArrays()
    {
        free(a); free(row_ind); free(col_ind);
        free(thread_j_s); free(thread_j_e);
        #ifdef PRINT_STATISTICS
            free(thread_time_barrier); free(thread_time_compute);
        #endif
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

void compute_sddmm(COOZSegmentedArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);


// ============================================================
// Kernels
// ============================================================


// NO ATOMIC required because threads own disjoint output rows.
// We just use standard += accumulation.
__attribute__((hot))
static inline
void
subkernel_val_coo_vec_noatomic(COOZSegmentedArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
{
    long r = coo->row_ind[j];
    long c_idx = coo->col_ind[j];
    ValueType val = coo->a[j];

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

void
COOZSegmentedArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;
    
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];

        // 1. Safe Initialization (No Race Condition)
        // Since we snapped boundaries to rows, we know exactly which rows we own.
        // But since 'row_ind' is now shuffled Z-order inside the segment, we can't just say "row_ind[j_s] to row_ind[j_e]".
        // We must scan the segment to find min/max row, OR (faster) rely on the original partition logic logic.
        // Actually, scanning is safest because Z-order might put the min row in the middle of the array.
        
        long min_r = -1, max_r = -1;
        if (j_e > j_s) {
            min_r = row_ind[j_s];
            max_r = row_ind[j_s];
            for(long j = j_s; j < j_e; j++) {
                if (row_ind[j] < min_r) min_r = row_ind[j];
                if (row_ind[j] > max_r) max_r = row_ind[j];
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
        
        // 2. Compute
        for (long j = j_s; j < j_e; j++)
        {
            #ifdef CUSTOM_COO_VEC_Z_ORDER_PREFETCH
                // Prefetch future X and Y locations into the cache
                // if (j + prefetch_distance < j_e) {
                    long future_r = row_ind[j + prefetch_distance];
                    long future_c = col_ind[j + prefetch_distance];
                    
                    // Prefetch Y for writing (1), high locality (3)
                    __builtin_prefetch(&y[future_r * k], 1, 3);
                    // Prefetch X for reading (0), high locality (3)
                    __builtin_prefetch(&x[future_c * k], 0, 3);
                // }
            #endif
            subkernel_val_coo_vec_noatomic(this, x, y, j, k);
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
COOZSegmentedArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
	compute_sddmm(this, x, y, out, k);
}

void
compute_sddmm(COOZSegmentedArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, __attribute__((unused)) int k)
{
	__attribute__((unused)) const ValueType alpha = 1.0;
	__attribute__((unused)) const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
		csr->y = y;
	}

	if (csr->out == NULL)
	{
		csr->out = out;
	}
}

// Factory function
struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COOZSegmentedArrays * coo = new COOZSegmentedArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    #ifdef CUSTOM_COO_VEC_Z_ORDER
        coo->format_name = (char *) "COO_Zorder";
    #elif CUSTOM_COO_VEC_Z_ORDER_PREFETCH
        coo->format_name = (char *) "COO_Zorder_Prefetch";
    #endif
    return coo;
}

void COOZSegmentedArrays::statistics_start() {
    int num_threads = omp_get_max_threads();
    num_loops = 0;
    for (long i=0; i<num_threads; i++) {
        thread_time_compute[i] = 0;
        thread_time_barrier[i] = 0;
    }
}
int COOZSegmentedArrays::statistics_print_data(char * buf, long buf_n) { return 0; }