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
// Same outer loop shape as subkernel_csr_vec_xrow_blocked in kernel_csr_vec_k_block_l1.cpp:
//
//     for (kb = 0; kb < k; kb += block_size) {
//         k_end = min(kb + block_size, k);
//         for (each owned row i) {
//             zero y[i, kb:k_end];
//             for (each non-zero j of row i)
//                 accumulate into y[i, kb:k_end];
//         }
//     }
//
// but with two changes. First, the per-non-zero accumulation for one block is
// subkernel_row_reg_fixedk from kernel_coo_vec_row_reg_fixedk.cpp instead of that file's
// subkernel_row_csr_vec_xrow_blocked: the CSR version accumulates straight into y[] in memory on
// every non-zero (same "load y, fmadd, store y" pattern kernel_coo_vec_row_reg.cpp's header
// explains is wasteful); this version keeps the block's accumulator in real vector registers for
// the whole row and only writes y[] once per row per block. Second, there's an
// '#pragma omp barrier' ("syncthreads") after every block: all threads finish sweeping their own
// rows for block b before any of them is allowed to start block b+1, so system-wide, at any
// instant, every thread is reading from the same block_size-wide slice of columns of x[] --
// bounding the combined working set pulled through the shared cache at any moment to (distinct
// columns touched) x block_size, instead of (distinct columns touched) x k. Exactly the same idea
// as fused_attention_coo_vector_blocking.cpp's d-blocking (see that file for the fuller writeup)
// -- and exactly as true there, the barrier is NOT required for correctness (rows stay
// thread-private throughout, so nothing changes about *what* gets computed).
//
// block_size is read once (from the COO_K_BLOCK_SIZE environment variable set in run.sh) and must
// be one of the sizes subkernel_row_reg_fixedk is specialized for below 512 (32/64/128/256) --
// deliberately excluding 512 itself, since that's "the whole row in one shot" territory that
// kernel_coo_vec_row_reg_fixedk.cpp's exact-match dispatch already covers with no blocking at all.
// The last block (k_end - kb < block_size, whenever k isn't a multiple of block_size, same as the
// CSR reference's k_end = min(kb+block_size,k)) falls back to a runtime-width accumulator instead
// of a register-blocked one, exactly like subkernel_row_csr_vec_xrow_blocked's own k_end handles
// an odd-sized last block by just running the same code with a narrower [kb,k_end) range.
//

struct COOArrays : Matrix_Format
{
    INT_T * row_ind; // Explicit row indices (of size nnz)
    INT_T * col_ind; // The colidx of each NNZ (of size nnz)
    ValueType * a;   // The values (of size NNZ)

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    int k_block_size;

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

        char * env_block_size = getenv("COO_K_BLOCK_SIZE");
        k_block_size = env_block_size ? atoi(env_block_size) : 0;
        if (k_block_size != 32 && k_block_size != 64 && k_block_size != 128 && k_block_size != 256) {
            fprintf(stderr, "Warning: COO_K_BLOCK_SIZE not set (or not one of 32/64/128/256); defaulting to 128\n");
            k_block_size = 128;
        }

        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));

        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                // Snap the start of each thread range to the next row boundary, so that no row
                // is ever split across threads.
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
            printf("k_block_size: %d (k: %d)\n", k_block_size, k);
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
void compute_coo_vector_xrow_fixedk_block(COOArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
COOArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;
    compute_coo_vector_xrow_fixedk_block(this, x, y, k);
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
    coo->format_name = (char *) "COO_RowSplit_Vec_FixedK_Block";
    return coo;
}

//==========================================================================================================================================
//= Subkernels COO
//==========================================================================================================================================

// True register-blocked accumulation of a NUM_CHUNKS*VEC_LEN-wide slice of one row, starting at
// k_offset within the row (row stride is still the real k). Identical to subkernel_row_reg_fixedk
// in kernel_coo_vec_row_reg_fixedk.cpp -- see that file for the full explanation of why acc[] ends
// up living in real vector registers instead of memory.
template <int NUM_CHUNKS>
__attribute__((hot))
static inline
void
subkernel_row_reg_fixedk(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j_s, long j_e, long row, int k, int k_offset = 0)
{
    vec_t(VTF, VEC_LEN) acc[NUM_CHUNKS];

    for (int i = 0; i < NUM_CHUNKS; i++)
        acc[i] = vec_set1(VTF, VEC_LEN, 0);

    for (long j = j_s; j < j_e; j++)
    {
        long c_idx = coo->col_ind[j];
        ValueType val = coo->a[j];
        vec_t(VTF, VEC_LEN) v_val = vec_set1(VTF, VEC_LEN, val);
        const ValueType * restrict x_row = &x[c_idx * k + k_offset];

        for (int i = 0; i < NUM_CHUNKS; i++)
        {
            vec_t(VTF, VEC_LEN) v_x = vec_loadu(VTF, VEC_LEN, &x_row[i * VEC_LEN]);
            acc[i] = vec_fmadd(VTF, VEC_LEN, v_val, v_x, acc[i]);
        }
    }

    ValueType * restrict y_row = &y[row * k + k_offset];
    for (int i = 0; i < NUM_CHUNKS; i++)
        vec_storeu(VTF, VEC_LEN, &y_row[i * VEC_LEN], acc[i]);
}

// Walks ONE thread's whole [j_s, j_e) non-zero range for a single k-block: splits it into
// contiguous per-row runs (COO entries are sorted by row) and register-blocks each one at the
// given k_offset. Called identically -- same NUM_CHUNKS, same k_offset -- by every thread for a
// given block, so every thread touches the same slice of x[] columns while this runs.
template <int NUM_CHUNKS>
__attribute__((hot))
static inline
void
subkernel_thread_block(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j_s, long j_e, int k, int k_offset)
{
    long j = j_s;
    while (j < j_e)
    {
        long row = coo->row_ind[j];
        long row_j_s = j;
        while (j < j_e && coo->row_ind[j] == row)
            j++;
        long row_j_e = j;

        subkernel_row_reg_fixedk<NUM_CHUNKS>(coo, x, y, row_j_s, row_j_e, row, k, k_offset);
    }
}

// Runtime-width accumulator for the last block, when k isn't a multiple of block_size (width =
// k_end - kb < block_size, same role as the CSR reference's k_end clamp). Still all rows of a
// given thread go through this together before the caller's barrier, so it's still synchronized
// the same way as a register-blocked block -- there's just no NUM_CHUNKS template for an
// arbitrary odd width, so this uses a plain runtime-sized local buffer instead.
__attribute__((hot))
static inline
void
subkernel_thread_tail(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j_s, long j_e, int k, int k_offset, int tail_width)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);
    c_e_vector = tail_width & mask;

    long j = j_s;
    while (j < j_e)
    {
        long row = coo->row_ind[j];
        long row_j_s = j;
        while (j < j_e && coo->row_ind[j] == row)
            j++;
        long row_j_e = j;

        ValueType local_buf[tail_width];
        for (c = 0; c < tail_width; c++)
            local_buf[c] = 0;

        for (long jj = row_j_s; jj < row_j_e; jj++)
        {
            long c_idx = coo->col_ind[jj];
            ValueType val = coo->a[jj];
            vec_t(VTF, VEC_LEN) v_val, v_x, v_prod, v_out;
            v_val = vec_set1(VTF, VEC_LEN, val);

            for (c = 0; c < c_e_vector; c += VEC_LEN)
            {
                v_out = vec_loadu(VTF, VEC_LEN, &local_buf[c]);
                v_x = vec_loadu(VTF, VEC_LEN, &x[c_idx * k + k_offset + c]);
                v_prod = vec_fmadd(VTF, VEC_LEN, v_val, v_x, v_out);
                vec_storeu(VTF, VEC_LEN, &local_buf[c], v_prod);
            }
            for (c = c_e_vector; c < tail_width; c++) {
                local_buf[c] += val * x[c_idx * k + k_offset + c];
            }
        }

        for (c = 0; c < tail_width; c++)
            y[row * k + k_offset + c] = local_buf[c];
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
compute_coo_vector_xrow_fixedk_block(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    int block_size = coo->k_block_size;

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
            // No nnz assigned to this thread (e.g. nnz < num_threads); the loops below simply
            // won't execute, but this thread still has to take part in every barrier below since
            // the number of blocks (derived from k and block_size) is the same for every thread.
            start_row = 0;
            end_row = -1;
        }

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        // for (kb = 0; kb < k; kb += block_size) -- same outer loop as
        // subkernel_csr_vec_xrow_blocked in kernel_csr_vec_k_block_l1.cpp.
        for (int kb = 0; kb < k; kb += block_size)
        {
            int k_end = (kb + block_size > k) ? k : kb + block_size;
            int width = k_end - kb;

            // Zero this block's slice for every row this thread owns (same as the CSR
            // reference's "for (c = kb; c < k_end; c++) y[i*k+c] = 0;", just once per row here
            // rather than inside subkernel_thread_block's own per-row loop).
            for (long i = start_row; i <= end_row; i++)
                for (long c = kb; c < k_end; c++)
                    y[i * k + c] = 0;

            // Sweep every row this thread owns, register-blocking each row's non-zeros over
            // [kb, k_end) -- subkernel_row_reg_fixedk from kernel_coo_vec_row_reg_fixedk.cpp
            // instead of subkernel_row_csr_vec_xrow_blocked's per-non-zero memory accumulate.
            // width == block_size for every block except possibly the last one.
            if (width == block_size) {
                switch (block_size)
                {
                    case 16:  subkernel_thread_block<16  / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                    case 32:  subkernel_thread_block<32  / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                    case 64:  subkernel_thread_block<64  / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                    case 128: subkernel_thread_block<128 / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                    case 256: subkernel_thread_block<256 / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                    case 512: subkernel_thread_block<512 / VEC_LEN>(coo, x, y, j_s, j_e, k, kb); break;
                }
            } else {
                // Last, narrower block (k not a multiple of block_size) -- same role as the CSR
                // reference's k_end clamp, just needing a runtime-width accumulator here since
                // there's no register-blocked template for an arbitrary odd width.
                subkernel_thread_tail(coo, x, y, j_s, j_e, k, kb, width);
            }

            // syncthreads(): every thread finishes block kb (columns [kb,k_end) of x[]) before
            // any of them is allowed to start block kb+1.
            #pragma omp barrier
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
