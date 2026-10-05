#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include <omp.h>

#include "macros/cpp_defines.h"

#include "bench_common.h"
#include "kernel_attention.h"

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
// This kernel fuses SDDMM-like score computation (Q @ K^T restricted to the attention mask
// non-zeros), online (streaming) softmax normalization, and SpMM-like accumulation of P @ V into
// a single pass over the COO non-zeros of the attention mask, following the same denominator
// composition rule that FlashAttention uses to avoid materializing the full N x N score matrix:
//
//     D_(A union {j}) = D_A * exp(m_A - m_(A union {j})) + exp(l_j - m_(A union {j}))
//     O_(A union {j}) = O_A * exp(m_A - m_(A union {j})) + exp(l_j - m_(A union {j})) * V_j
//
// Here the running set A grows one non-zero (one key/value column) at a time instead of one tile
// at a time, since on CPU there is no shared-memory tiling stage: the vectorization granularity is
// simply the head dimension d, exactly like the x-row vectorization already used for SpMM in
// kernel_coo_vec.cpp. Only the CUSTOM_COO_VEC_XROW_ROW_SPLIT partitioning scheme is used here
// (each thread owns a contiguous, non-atomic range of COO non-zeros whose boundaries are snapped
// to row boundaries), since the online-softmax recurrence above has a sequential dependency along
// the non-zeros of a given row and therefore cannot be split across threads without a reduction.
//

struct FusedAttentionCOOArrays : Matrix_Format_Attention
{
    INT_T * row_ind; // Explicit row indices (of size nnz), i.e. the query index of each mask entry.
    INT_T * col_ind; // The column index of each mask entry, i.e. the key/value index it attends to.
    ValueType * a;   // The mask values (of size nnz); unused in the score itself (pattern-only mask).

    ValueType * m_state = NULL; // Running max of the scores seen so far per query row (size m).
    ValueType * D_state = NULL; // Running softmax denominator per query row (size m).

    long num_loops;

    FusedAttentionCOOArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int d)
        : Matrix_Format_Attention(m, n, nnz, d)
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

        m_state = (ValueType *) malloc(m * sizeof(*m_state));
        D_state = (ValueType *) malloc(m * sizeof(*D_state));

        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));

        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                // CUSTOM_COO_VEC_XROW_ROW_SPLIT: snap the start of each thread range to the next row
                // boundary, so that no row (and thus no online-softmax accumulation chain) is split
                // across threads.
                if (tnum > 0 && thread_j_s[tnum] < nnz) {
                    while (thread_j_s[tnum] < nnz && row_ind[thread_j_s[tnum]] == row_ind[thread_j_s[tnum] - 1]) {
                        thread_j_s[tnum]++;
                    }
                }

                _Pragma("omp barrier")

                // Set end based on neighbor start.
                if (tnum == num_threads - 1) {
                    thread_j_e[tnum] = nnz;
                } else {
                    thread_j_e[tnum] = thread_j_s[tnum + 1];
                }

                // Safety check: if a single row is massive, a thread might have start > end.
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

    ~FusedAttentionCOOArrays()
    {
        free(a);
        free(row_ind);
        free(col_ind);
        free(m_state);
        free(D_state);
        free(thread_j_s);
        free(thread_j_e);

        #ifdef PRINT_STATISTICS
            free(thread_time_barrier);
            free(thread_time_compute);
        #endif
    }

    void attention(ValueType * Q, ValueType * K, ValueType * V, ValueType * O, int d);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

// Forward declarations
void compute_coo_attention_xrow(FusedAttentionCOOArrays * restrict attn, ValueType * restrict Q, ValueType * restrict K, ValueType * restrict V, ValueType * restrict O, int d);

void
FusedAttentionCOOArrays::attention(ValueType * Q, ValueType * K, ValueType * V, ValueType * O, int d)
{
    num_loops++;
    compute_coo_attention_xrow(this, Q, K, V, O, d);
}

struct Matrix_Format_Attention *
csr_to_format_attention(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int d)
{
    struct FusedAttentionCOOArrays * attn = new FusedAttentionCOOArrays(row_ptr, col_ind, values, m, n, nnz, d);
    attn->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T)) + 2 * m * sizeof(ValueType);
    attn->format_name = (char *) "FusedAttention_COO_RowSplit_Vec";
    return attn;
}

//==========================================================================================================================================
//= Subkernels COO (vectorized over the head dimension d, same style as kernel_coo_vec.cpp)
//==========================================================================================================================================

// Q_row . K_row, vectorized reduction over d (same structure as subkernel_val_coo_sddmm).
__attribute__((hot))
static inline
ValueType
subkernel_dot_product(ValueType * restrict Q_row, ValueType * restrict K_row, int d)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_q, v_k, v_sum;
    c_e_vector = d & mask;

    v_sum = vec_set1(VTF, VEC_LEN, 0);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_q = vec_loadu(VTF, VEC_LEN, &Q_row[c]);
        v_k = vec_loadu(VTF, VEC_LEN, &K_row[c]);
        v_sum = vec_fmadd(VTF, VEC_LEN, v_q, v_k, v_sum);
    }

    ValueType dot = vec_reduce_add(VTF, VEC_LEN, v_sum);

    for (c = c_e_vector; c < d; c++) {
        dot += Q_row[c] * K_row[c];
    }

    return dot;
}

// O_row = O_row * correction + p * V_row, vectorized over d
// (same load/fmadd/store shape as subkernel_val_coo_vec_xrow_noatomic, with an extra rescale of the accumulator).
__attribute__((hot))
static inline
void
subkernel_rescale_accumulate(ValueType * restrict O_row, ValueType * restrict V_row, ValueType correction, ValueType p, int d)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_o, v_v, v_corr, v_p;
    c_e_vector = d & mask;

    v_corr = vec_set1(VTF, VEC_LEN, correction);
    v_p    = vec_set1(VTF, VEC_LEN, p);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_o = vec_loadu(VTF, VEC_LEN, &O_row[c]);
        v_v = vec_loadu(VTF, VEC_LEN, &V_row[c]);
        v_o = vec_mul(VTF, VEC_LEN, v_o, v_corr);
        v_o = vec_fmadd(VTF, VEC_LEN, v_p, v_v, v_o);
        vec_storeu(VTF, VEC_LEN, &O_row[c], v_o);
    }

    for (c = c_e_vector; c < d; c++) {
        O_row[c] = O_row[c] * correction + p * V_row[c];
    }
}

// O_row *= inv_D, vectorized over d (final softmax normalization).
__attribute__((hot))
static inline
void
subkernel_normalize_row(ValueType * restrict O_row, ValueType inv_D, int d)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_o, v_invD;
    c_e_vector = d & mask;

    v_invD = vec_set1(VTF, VEC_LEN, inv_D);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_o = vec_loadu(VTF, VEC_LEN, &O_row[c]);
        v_o = vec_mul(VTF, VEC_LEN, v_o, v_invD);
        vec_storeu(VTF, VEC_LEN, &O_row[c], v_o);
    }

    for (c = c_e_vector; c < d; c++) {
        O_row[c] *= inv_D;
    }
}

// Processes a single (query, key) pair, i.e. one non-zero of the attention mask:
// score = scale * Q[r,:] . K[c_idx,:]; online-softmax update of m_state[r], D_state[r], O[r,:].
__attribute__((hot))
static inline
void
subkernel_val_coo_attention_xrow_noatomic(FusedAttentionCOOArrays * restrict attn, ValueType * restrict Q, ValueType * restrict K, ValueType * restrict V, ValueType * restrict O, long j, int d, ValueType scaling)
{
    long r = attn->row_ind[j];
    long c_idx = attn->col_ind[j];

    ValueType * Q_row = &Q[r * d];
    ValueType * K_row = &K[c_idx * d];
    ValueType * V_row = &V[c_idx * d];
    ValueType * O_row = &O[r * d];

    ValueType score = scaling * subkernel_dot_product(Q_row, K_row, d);

    ValueType m_old = attn->m_state[r];
    ValueType m_new = (score > m_old) ? score : m_old;

    // D_(A union {j}) = D_A * exp(m_A - m_(A union {j})) + exp(l_j - m_(A union {j}))   (composition rule, |B| = 1)
    ValueType correction = std::exp(m_old - m_new);
    ValueType p = std::exp(score - m_new);

    // O_(A union {j}) = O_A * exp(m_A - m_(A union {j})) + p * V_j
    subkernel_rescale_accumulate(O_row, V_row, correction, p, d);

    attn->D_state[r] = attn->D_state[r] * correction + p;
    attn->m_state[r] = m_new;
}

//==========================================================================================================================================
//= COO Main Computation Kernel
//==========================================================================================================================================

void
compute_coo_attention_xrow(FusedAttentionCOOArrays * restrict attn, ValueType * restrict Q, ValueType * restrict K, ValueType * restrict V, ValueType * restrict O, int d)
{
    ValueType scaling = (ValueType) (1.0 / sqrt((double) d));
    ValueType neg_inf = -std::numeric_limits<ValueType>::infinity();

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s, j_e, start_row, end_row;
        j_s = thread_j_s[tnum];
        j_e = thread_j_e[tnum];
        if (j_e > j_s) {
            start_row = attn->row_ind[j_s];
            end_row = attn->row_ind[j_e - 1];
        } else {
            // No nnz assigned to this thread (e.g. nnz < num_threads); skip init/compute/normalize below.
            start_row = 0;
            end_row = -1;
        }

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        // Initialize O, D and m for the query rows owned by this thread.
        for (long i = start_row; i <= end_row; i++)
        {
            attn->m_state[i] = neg_inf;
            attn->D_state[i] = 0;
            for (long c = 0; c < d; c++)
                O[i * d + c] = 0;
        }

        for (long j = j_s; j < j_e; j++)
        {
            subkernel_val_coo_attention_xrow_noatomic(attn, Q, K, V, O, j, d, scaling);
        }

        // Finalize: O[r,:] /= D[r] (rows with no non-zeros, i.e. D == 0, are left as all-zero).
        for (long i = start_row; i <= end_row; i++)
        {
            if (attn->D_state[i] > 0)
            {
                ValueType inv_D = ((ValueType) 1) / attn->D_state[i];
                subkernel_normalize_row(&O[i * d], inv_D, d);
            }
        }

        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1, _Pragma("omp barrier"));
        thread_time_barrier[tnum] += time;
        #endif
    }
}

//==========================================================================================================================================
//= Statistics
//==========================================================================================================================================

void
FusedAttentionCOOArrays::statistics_start()
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
FusedAttentionCOOArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}
