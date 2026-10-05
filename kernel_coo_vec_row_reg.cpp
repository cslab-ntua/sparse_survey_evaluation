#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

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

//
// Register-blocked variant of the COO_RowSplit_Vec kernel (kernel_coo_vec.cpp).
//
// subkernel_val_coo_vec_xrow_noatomic there does, per non-zero:
//   v_y = load(y[row])   v_x = load(x[col])   v_y = fmadd(val,v_x,v_y)   store(y[row], v_y)
// i.e. it loads AND stores the *entire output row* from/to memory on every single non-zero
// touching that row. For a row of degree d, that's d loads + d stores of a k-float row, even
// though the row's running total never needed to leave the CPU in between.
//
// This version instead keeps that running total in a small local buffer (which the compiler is
// free to keep in registers/L1, exactly like FusedMM's generated SpMM kernel keeps the whole
// output row in vector registers -- see sgfusedMM_K*_spmm_csr.c) for as long as we're processing
// non-zeros of the same row, and only writes to y[] once, when we move on to the next row. Since
// the COO entries are sorted by row and each thread's non-zero range is snapped to row boundaries
// (CUSTOM_COO_VEC_XROW_ROW_SPLIT, same as kernel_coo_vec.cpp), no row is ever split across
// threads, so this "flush on row change" is always safe.
//

struct COOArrays : Matrix_Format
{
    INT_T * row_ind; // Explicit row indices (of size nnz)
    INT_T * col_ind; // The colidx of each NNZ (of size nnz)
    ValueType * a;   // The values (of size NNZ)

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    long num_loops;

    COOArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k)
        : Matrix_Format(m, n, nnz, k)
    {
        int num_threads = omp_get_max_threads();
        double time_balance;

        row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
        col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
        a = (ValueType *) malloc(nnz * sizeof(*a));

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

        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                // Snap the start of each thread range to the next row boundary, so that no row
                // (and thus no row-local accumulator) is split across threads.
                if (tnum > 0 && thread_j_s[tnum] < nnz) {
                    while (thread_j_s[tnum] < nnz && row_ind[thread_j_s[tnum]] == row_ind[thread_j_s[tnum] - 1]) {
                        thread_j_s[tnum]++;
                    }
                }

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

    ~COOArrays()
    {
        free(a);
        free(row_ind);
        free(col_ind);
        free(thread_j_s);
        free(thread_j_e);

        #ifdef PRINT_STATISTICS
            free(thread_time_barrier);
            free(thread_time_compute);
        #endif
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

// Forward declarations
void compute_coo_vector_xrow_row_reg(COOArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
COOArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;
    compute_coo_vector_xrow_row_reg(this, x, y, k);
}

void
COOArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    compute_coo_sddmm(this, x, y, out, k);
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COOArrays * coo = new COOArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    coo->format_name = (char *) "COO_RowSplit_Vec_RowReg";
    return coo;
}

//==========================================================================================================================================
//= Subkernels COO
//==========================================================================================================================================

// out_buf[c] += val[j] * x[col_ind[j], c], vectorized over k. Accumulates into a caller-supplied
// local buffer instead of the shared output matrix, so the same subkernel can be reused for every
// non-zero of a row without ever touching y[] in between.
__attribute__((hot))
static inline
void
subkernel_val_coo_vec_xrow_partial(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict out_buf, long j, int k)
{
    long c_idx = coo->col_ind[j];
    ValueType val = coo->a[j];

    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_val, v_x, v_prod, v_out;

    c_e_vector = k & mask;
    v_val = vec_set1(VTF, VEC_LEN, val);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_out = vec_loadu(VTF, VEC_LEN, &out_buf[c]);
        v_x = vec_loadu(VTF, VEC_LEN, &x[c_idx * k + c]);
        v_prod = vec_fmadd(VTF, VEC_LEN, v_val, v_x, v_out);
        vec_storeu(VTF, VEC_LEN, &out_buf[c], v_prod);
    }

    for (c = c_e_vector; c < k; c++) {
        out_buf[c] += val * x[c_idx * k + c];
    }
}

__attribute__((hot))
static inline
void
subkernel_val_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, long j, int k)
{
    long r = coo->row_ind[j];
    long c_idx = coo->col_ind[j];
    ValueType val = coo->a[j];

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

//==========================================================================================================================================
//= COO Main Computation Kernel
//==========================================================================================================================================

void
compute_coo_vector_xrow_row_reg(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s, j_e, start_row, end_row;
        j_s = thread_j_s[tnum];
        j_e = thread_j_e[tnum];
        if (j_e > j_s) {
            start_row = coo->row_ind[j_s];
            end_row = coo->row_ind[j_e - 1];
        } else {
            // No nnz assigned to this thread (e.g. nnz < num_threads); skip zero-init/compute below.
            start_row = 0;
            end_row = -1;
        }

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        // Zero every row this thread owns up front, so rows with zero non-zeros still end up all-zero
        // (the row-local accumulation loop below never visits them, since they have no COO entries).
        for (long i = start_row; i <= end_row; i++) {
            for (long c = 0; c < k; c++)
                y[i * k + c] = 0;
        }

        // Register-blocked accumulation: keep the running total for the current row in a small
        // local buffer (which the compiler can keep in registers/L1 across the whole row, exactly
        // like FusedMM's generated SpMM kernel keeps its output row in vector registers) and flush
        // it to y[] exactly once per row -- when we notice the row has changed -- instead of once
        // per non-zero.
        ValueType local_buf[k];
        long cur_row = -1;

        for (long j = j_s; j < j_e; j++)
        {
            long r = coo->row_ind[j];
            if (r != cur_row) {
                if (cur_row != -1) {
                    for (long c = 0; c < k; c++)
                        y[cur_row * k + c] = local_buf[c];
                }
                for (long c = 0; c < k; c++)
                    local_buf[c] = 0;
                cur_row = r;
            }
            subkernel_val_coo_vec_xrow_partial(coo, x, local_buf, j, k);
        }
        // Flush the last row this thread touched.
        if (cur_row != -1) {
            for (long c = 0; c < k; c++)
                y[cur_row * k + c] = local_buf[c];
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
compute_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k)
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
            subkernel_val_coo_sddmm(coo, x, y, out, j, k);
        }
    }
}

//==========================================================================================================================================
//= Statistics
//==========================================================================================================================================

void
COOArrays::statistics_start()
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
COOArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}
