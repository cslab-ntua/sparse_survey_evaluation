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
// Blocked variant of fused_attention_coo_vector.cpp.
//
// The original kernel streams over the COO non-zeros of a row and, for every single non-zero,
// touches the *entire* head dimension d of that non-zero's K row and V row (see
// subkernel_dot_product / subkernel_rescale_accumulate in fused_attention_coo_vector.cpp). Since
// the non-zeros of a row generally point at unrelated, scattered key/value columns, the set of K/V
// cache lines touched while a thread works through a row's non-zeros is effectively "all of K and
// all of V, d elements at a time, in whatever column order the mask happens to list them" -- no
// control over how much of K/V is resident at once.
//
// This version instead splits the head dimension d into blocks of 'd_block_size' elements
// (CUSTOM_ATTENTION_D_BLOCK_SIZE, read from the ATTENTION_D_BLOCK_SIZE environment variable set in
// run.sh) and restructures the single online-softmax pass into three phases, each swept over the
// whole non-zero range one d-block at a time:
//
//   Phase 1 (score):  for each d-block, accumulate the partial Q.K dot product of every non-zero
//                      into a persistent per-non-zero buffer. Touches only a d_block_size-wide
//                      slice of K per K row.
//   Phase 2 (softmax): pure scalar reduction over the completed per-row scores (row max, then
//                      exp-sum) -- no K or V access at all.
//   Phase 3 (output):  for each d-block, accumulate p * V into O. Touches only a d_block_size-wide
//                      slice of V per V row.
//
// A dot product is a reduction over the *entire* head dimension, so score can't be finalized until
// every d-block has been visited -- that's why this can no longer be a single online-softmax pass
// like the original (which folds one whole non-zero, all d elements at once, into the running
// max/denominator/output in one shot). Instead this is a classic two-pass (max, then exp-sum)
// softmax, split further into per-d-block sweeps.
//
// Each row is still owned exclusively by one thread (same CUSTOM_COO_VEC_XROW_ROW_SPLIT partition
// as the original), so nothing here is a data-race concern even without synchronization. The
// '#pragma omp barrier' placed between d-blocks is therefore not required for correctness -- it is
// there on purpose, so that at any point in time *every* thread is working on the same d-block:
// system-wide, the kernel only ever touches a d_block_size-wide slice of K (Phase 1) or V (Phase 3)
// across however many distinct columns the non-zeros in flight reference, instead of the full d
// columns of K/V being pulled through the shared cache simultaneously by all threads.
//

struct FusedAttentionCOOArraysBlocking : Matrix_Format_Attention
{
    INT_T * row_ind; // Explicit row indices (of size nnz), i.e. the query index of each mask entry.
    INT_T * col_ind; // The column index of each mask entry, i.e. the key/value index it attends to.
    ValueType * a;   // The mask values (of size nnz); unused in the score itself (pattern-only mask).

    ValueType * m_state = NULL; // Running (then final) max score per query row (size m).
    ValueType * D_state = NULL; // Running (then final) softmax denominator per query row (size m).
    ValueType * score = NULL;   // Per-non-zero scratch buffer (size nnz): raw dot product, then
                                 // scaled score, then unnormalized softmax weight p', reused in place
                                 // across the three phases below.

    int d_block_size;

    long num_loops;

    FusedAttentionCOOArraysBlocking(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int d)
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
        score = (ValueType *) malloc((nnz > 0 ? nnz : 1) * sizeof(*score));

        char * env_block_size = getenv("ATTENTION_D_BLOCK_SIZE");
        d_block_size = env_block_size ? atoi(env_block_size) : 0;
        if (d_block_size <= 0) {
            fprintf(stderr, "Warning: ATTENTION_D_BLOCK_SIZE not set (or invalid); defaulting to 16\n");
            d_block_size = 16;
        }
        if (d_block_size > d)
            d_block_size = d;

        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));

        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                // CUSTOM_COO_VEC_XROW_ROW_SPLIT: snap the start of each thread range to the next row
                // boundary, so that no row (and thus no per-row score/softmax/output state) is split
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
            printf("d_block_size: %d (d: %d)\n", d_block_size, d);
        #endif
    }

    ~FusedAttentionCOOArraysBlocking()
    {
        free(a);
        free(row_ind);
        free(col_ind);
        free(m_state);
        free(D_state);
        free(score);
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
void compute_coo_attention_xrow_blocked(FusedAttentionCOOArraysBlocking * restrict attn, ValueType * restrict Q, ValueType * restrict K, ValueType * restrict V, ValueType * restrict O, int d);

void
FusedAttentionCOOArraysBlocking::attention(ValueType * Q, ValueType * K, ValueType * V, ValueType * O, int d)
{
    num_loops++;
    compute_coo_attention_xrow_blocked(this, Q, K, V, O, d);
}

struct Matrix_Format_Attention *
csr_to_format_attention(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int d)
{
    struct FusedAttentionCOOArraysBlocking * attn = new FusedAttentionCOOArraysBlocking(row_ptr, col_ind, values, m, n, nnz, d);
    attn->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T)) + 2 * m * sizeof(ValueType) + nnz * sizeof(ValueType);
    attn->format_name = (char *) "FusedAttention_COO_RowSplit_Vec_Blocked";
    return attn;
}

//==========================================================================================================================================
//= Subkernels COO (vectorized over a [c_start, c_end) slice of the head dimension d)
//==========================================================================================================================================

// Partial Q_row . K_row over [c_start, c_end) only (same vectorized-reduction shape as
// subkernel_dot_product in fused_attention_coo_vector.cpp, but restricted to one d-block; mirrors
// subkernel_row_csr_vec_xrow_blocked's k_start/k_end handling in kernel_csr_vec_k_block_l1.cpp).
__attribute__((hot))
static inline
ValueType
subkernel_dot_product_block(ValueType * restrict Q_row, ValueType * restrict K_row, int c_start, int c_end)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_q, v_k, v_sum;
    int chunk = c_end - c_start;
    c_e_vector = c_start + (chunk & mask);

    v_sum = vec_set1(VTF, VEC_LEN, 0);

    for (c = c_start; c < c_e_vector; c += VEC_LEN)
    {
        v_q = vec_loadu(VTF, VEC_LEN, &Q_row[c]);
        v_k = vec_loadu(VTF, VEC_LEN, &K_row[c]);
        v_sum = vec_fmadd(VTF, VEC_LEN, v_q, v_k, v_sum);
    }

    ValueType dot = vec_reduce_add(VTF, VEC_LEN, v_sum);

    for (c = c_e_vector; c < c_end; c++) {
        dot += Q_row[c] * K_row[c];
    }

    return dot;
}

// O_row[c_start:c_end] += p * V_row[c_start:c_end]. No rescale term is needed here (unlike the
// online version's subkernel_rescale_accumulate): p is already the final unnormalized softmax
// weight by the time Phase 3 runs, since Phase 1+2 finalize every non-zero's score before any
// output accumulation begins.
__attribute__((hot))
static inline
void
subkernel_accumulate_block(ValueType * restrict O_row, ValueType * restrict V_row, ValueType p, int c_start, int c_end)
{
    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_o, v_v, v_p;
    int chunk = c_end - c_start;
    c_e_vector = c_start + (chunk & mask);

    v_p = vec_set1(VTF, VEC_LEN, p);

    for (c = c_start; c < c_e_vector; c += VEC_LEN)
    {
        v_o = vec_loadu(VTF, VEC_LEN, &O_row[c]);
        v_v = vec_loadu(VTF, VEC_LEN, &V_row[c]);
        v_o = vec_fmadd(VTF, VEC_LEN, v_p, v_v, v_o);
        vec_storeu(VTF, VEC_LEN, &O_row[c], v_o);
    }

    for (c = c_e_vector; c < c_end; c++) {
        O_row[c] += p * V_row[c];
    }
}

// O_row *= inv_D, vectorized over the whole d (final softmax normalization; doesn't touch K or V,
// so there's no reason to block it).
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

//==========================================================================================================================================
//= COO Main Computation Kernel (d-blocked, three phases, thread-synchronized between blocks)
//==========================================================================================================================================

void
compute_coo_attention_xrow_blocked(FusedAttentionCOOArraysBlocking * restrict attn, ValueType * restrict Q, ValueType * restrict K, ValueType * restrict V, ValueType * restrict O, int d)
{
    ValueType scaling = (ValueType) (1.0 / sqrt((double) d));
    ValueType neg_inf = -std::numeric_limits<ValueType>::infinity();
    int block_size = attn->d_block_size;

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
            // No nnz assigned to this thread (e.g. nnz < num_threads); the loops below simply
            // won't execute, but this thread still has to take part in every '#pragma omp barrier'
            // below since the number of d-blocks is the same for every thread.
            start_row = 0;
            end_row = -1;
        }

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        // Initialize per-row state for the query rows owned by this thread.
        for (long i = start_row; i <= end_row; i++) {
            attn->m_state[i] = neg_inf;
            attn->D_state[i] = 0;
        }
        for (long j = j_s; j < j_e; j++) {
            attn->score[j] = 0;
        }

        // Phase 1: raw Q.K dot product, one d-block at a time. Only a block_size-wide slice of K
        // is read per non-zero during any given iteration of this loop, and every thread is on the
        // same d-block at the same time (enforced by the barrier below).
        for (int c_s = 0; c_s < d; c_s += block_size) {
            int c_e = (c_s + block_size > d) ? d : c_s + block_size;

            for (long j = j_s; j < j_e; j++) {
                long r = attn->row_ind[j];
                long c_idx = attn->col_ind[j];
                attn->score[j] += subkernel_dot_product_block(&Q[r * d], &K[c_idx * d], c_s, c_e);
            }

            #pragma omp barrier
        }

        // Phase 2: finalize the (numerically stable) softmax weights from the now-complete scores.
        // Pure scalar work over this thread's own rows/non-zeros -- no K or V access, no barrier
        // needed (every row is owned by exactly one thread).

        // Pass A: scale the raw dot products and find the per-row max.
        for (long j = j_s; j < j_e; j++) {
            long r = attn->row_ind[j];
            ValueType s = scaling * attn->score[j];
            attn->score[j] = s;
            if (s > attn->m_state[r])
                attn->m_state[r] = s;
        }
        // Pass B: turn each score into its unnormalized softmax weight p', and sum into D_state.
        for (long j = j_s; j < j_e; j++) {
            long r = attn->row_ind[j];
            ValueType p = std::exp(attn->score[j] - attn->m_state[r]);
            attn->score[j] = p;
            attn->D_state[r] += p;
        }

        // Zero O for the rows owned by this thread, ready for Phase 3's accumulation.
        for (long i = start_row; i <= end_row; i++) {
            for (long c = 0; c < d; c++)
                O[i * d + c] = 0;
        }

        // Phase 3: O += p' * V, one d-block at a time. Only a block_size-wide slice of V is read
        // per non-zero during any given iteration, all threads again kept in lockstep by the
        // barrier so the whole system only ever touches one d-block of V at a time.
        for (int c_s = 0; c_s < d; c_s += block_size) {
            int c_e = (c_s + block_size > d) ? d : c_s + block_size;

            for (long j = j_s; j < j_e; j++) {
                long r = attn->row_ind[j];
                long c_idx = attn->col_ind[j];
                subkernel_accumulate_block(&O[r * d], &V[c_idx * d], attn->score[j], c_s, c_e);
            }

            #pragma omp barrier
        }

        // Finalize: O[r,:] /= D[r] (rows with no non-zeros, i.e. D == 0, are left as all-zero).
        for (long i = start_row; i <= end_row; i++) {
            if (attn->D_state[i] > 0) {
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
FusedAttentionCOOArraysBlocking::statistics_start()
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
FusedAttentionCOOArraysBlocking::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}
