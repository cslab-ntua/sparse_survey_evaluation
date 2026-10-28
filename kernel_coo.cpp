#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
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

//==========================================================================================================================================
//= Variant selection
//==========================================================================================================================================
//
// One file for all the vectorized COO SpMM variants. A variant is a base kernel + a reordering, chosen at compile time:
//
//   Base kernel (exactly one):
//     CUSTOM_COO_VEC_XROW_ROW_SPLIT            per-nnz y += val*x, thread ranges snapped to row boundaries     (coo_base_xrow.h)
//     CUSTOM_COO_VEC_XROW_ATOMIC               per-nnz with atomics, plain nnz-balanced ranges                 (coo_base_xrow.h)
//     CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE  per-nnz, plain ranges, boundary row in a private buffer         (coo_base_xrow.h)
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK            row runs accumulated in registers, fixed k, row-snapped ranges  (coo_base_row_reg_fixedk.h)
//
//   Modifiers:
//     CUSTOM_COO_VEC_XROW_COLIND0              every col_ind = csr_ja[0] (upper bound of x reuse), any base
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_ACCUMULATE force the fixedk ACCUMULATE mode even for row-sorted orders
//     CUSTOM_COO_VEC_PREFETCH                  prefetch the rows of the non-zero PREFETCH_DIST (env, default 4) positions ahead:
//                                              row_split / atomic prefetch its x and y rows, fixedk its x row (y is in registers)
//       CUSTOM_COO_VEC_PREFETCH_FULL_ROW       prefetch every cache line of those rows, not only the first one
//     CUSTOM_COO_VEC_ZERO_STATIC               zero all of y with one static (row-balanced) parallel loop before the kernel,
//                                              instead of every thread zeroing its own nnz-balanced row range
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE=<S>  fixedk: k-slicing, process every row in slices of S columns (S/8 accumulators
//                                              with AVX2; S = 120 is the largest slice without spills), one kernel for any k.
//                                              Default S = 128; S = 0 switches back to the per-k kernels (k = 16 .. 512)
//                                              and the generic path for other k
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER=<W>
//                                              fixedk: outer slicing, all runs of a thread for columns [0, W), then
//                                              [W, 2W), ... (cache blocking of x in k; W a multiple of 8)
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER_PACK
//                                              fixedk outer slicing: copy every strip of x into a contiguous n x W
//                                              buffer first (avoids cache-set conflicts for power-of-two k)
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_RUN_SCAN   fixedk: find the end of every run by scanning row_ind in the timed loop
//                                              (previous behaviour). Default: run boundaries precomputed once (like CSR
//                                              row pointers)
//     CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL   fixedk: previous zeroing (each thread zeroes its whole row range, ACCUMULATE
//                                              adds every run). Default: only the empty rows are zeroed, and the first run
//                                              of each row stores (see coo_base_row_reg_fixedk.h)
//
//   Reordering (at most one, default none):
//     COO_REORDER_RCM                          group A: Reverse Cuthill-McKee relabeling (coo_reorder_rcm.h)
//       CUSTOM_COO_VEC_RCM_PERMUTE_IN_TIMING   permute x / y on every spmm() call, inside the measured time
//     COO_REORDER_Z_ORDER                      group B: Z-order inside each thread segment (coo_reorder_z_order.h)
//     COO_REORDER_HILBERT                      group B: Hilbert curve order inside each thread segment (coo_reorder_hilbert.h)
//     COO_REORDER_CROSS                        group B: greedy row / column "cross" walk inside each segment (coo_reorder_cross.h)
//     COO_REORDER_FRONTIER                     group B: greedy scored-frontier walk inside each segment (coo_reorder_frontier.h)
//     COO_REORDER_RINGBUFFER                   group B: frontier walk with a ring-buffer cache model (coo_reorder_ringbuffer.h)
//     COO_REORDER_RINGBUFFER_ALL_SEED            ... with the frontier pre-filled from the seed rows and columns
//     COO_REORDER_RINGBUFFER_ROULETTE            ... with a stochastic (tiered / seed / random) choice and random eviction

#if (defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT) + defined(CUSTOM_COO_VEC_XROW_ATOMIC) + defined(CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE) + defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)) != 1
    #error "kernel_coo.cpp: define exactly one base kernel (CUSTOM_COO_VEC_XROW_ROW_SPLIT / _ATOMIC / _PERFECT_NNZ_BALANCE / CUSTOM_COO_VEC_ROW_REG_FIXEDK)"
#endif

#if defined(CUSTOM_COO_VEC_PREFETCH) && defined(CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_PREFETCH is not implemented for the perfect nnz balance base"
#endif

#if defined(CUSTOM_COO_VEC_PREFETCH_FULL_ROW) && !defined(CUSTOM_COO_VEC_PREFETCH)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_PREFETCH_FULL_ROW needs CUSTOM_COO_VEC_PREFETCH"
#endif

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE) && !defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE is for the fixedk base"
#endif

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER) && !defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER is for the fixedk base"
#endif

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER_PACK) && !defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER_PACK needs CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER=<W>"
#endif

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_RUN_SCAN) && !defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_ROW_REG_FIXEDK_RUN_SCAN is for the fixedk base"
#endif

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL) && (!defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK) || defined(CUSTOM_COO_VEC_ZERO_STATIC))
    #error "kernel_coo.cpp: CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL is for the fixedk base, and excludes CUSTOM_COO_VEC_ZERO_STATIC"
#endif

// Prefetch helper. __builtin_prefetch needs compile-time constant rw / locality arguments, so this is a macro.
// COO_PREFETCH_ROW(ptr, width, rw): the first cache line of the row starting at ptr, or with
// CUSTOM_COO_VEC_PREFETCH_FULL_ROW all the cache lines of its first width elements. Locality 3: keep in all cache levels.
// The Makefile sets LEVEL1_DCACHE_LINESIZE from getconf, which can be empty / 0 on some systems: fall back to 64 bytes.
#if !defined(LEVEL1_DCACHE_LINESIZE) || (LEVEL1_DCACHE_LINESIZE + 0) <= 0
    #define COO_CACHE_LINE_SIZE  64
#else
    #define COO_CACHE_LINE_SIZE  LEVEL1_DCACHE_LINESIZE
#endif
#define COO_PREFETCH_LINE_ELEMS  ((long) (COO_CACHE_LINE_SIZE / sizeof(ValueType)))
#ifdef CUSTOM_COO_VEC_PREFETCH_FULL_ROW
    #define COO_PREFETCH_ROW(ptr, width, rw)                                      \
        do {                                                                      \
            for (long _c = 0; _c < (long) (width); _c += COO_PREFETCH_LINE_ELEMS) \
                __builtin_prefetch(&(ptr)[_c], rw, 3);                            \
        } while (0)
#else
    #define COO_PREFETCH_ROW(ptr, width, rw)  __builtin_prefetch((ptr), rw, 3)
#endif

// Bases that need every row to be owned by exactly one thread snap the thread ranges to row boundaries.
#if defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT) || defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
    #define COO_PARTITION_ROW_SPLIT
#endif


INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;

// Rows touched by each thread, computed at partitioning time (before any reordering inside the segments).
// Empty threads get [0, -1].
INT_T * thread_row_s = NULL;
INT_T * thread_row_e = NULL;

ValueType * thread_v_e = NULL;

double * thread_time_compute, * thread_time_barrier;

long prefetch_distance = 4;   // CUSTOM_COO_VEC_PREFETCH, set from the PREFETCH_DIST env variable in the constructor

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

    COOArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k);
    ~COOArrays();

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

//==========================================================================================================================================
//= Reordering
//==========================================================================================================================================

#if defined(COO_REORDER_RCM)
    #include "coo_reorder_rcm.h"
#elif defined(COO_REORDER_Z_ORDER)
    #include "coo_reorder_z_order.h"
#elif defined(COO_REORDER_HILBERT)
    #include "coo_reorder_hilbert.h"
#elif defined(COO_REORDER_CROSS)
    #include "coo_reorder_cross.h"
#elif defined(COO_REORDER_FRONTIER)
    #include "coo_reorder_frontier.h"
#elif defined(COO_REORDER_RINGBUFFER) || defined(COO_REORDER_RINGBUFFER_ALL_SEED) || defined(COO_REORDER_RINGBUFFER_ROULETTE)
    #include "coo_reorder_ringbuffer.h"
#else
    #include "coo_reorder_none.h"
#endif

// The fixedk write mode follows from the reordering: STORE only if every row stays a single run.
#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_ACCUMULATE) || !COO_REORDER_KEEPS_ROWS
    #define COO_FIXEDK_ACCUMULATE  1
#else
    #define COO_FIXEDK_ACCUMULATE  0
#endif

//==========================================================================================================================================
//= Base kernels
//==========================================================================================================================================

#if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
    #include "coo_base_row_reg_fixedk.h"
#else
    #include "coo_base_xrow.h"
#endif

//==========================================================================================================================================
//= Constructor
//==========================================================================================================================================

COOArrays::COOArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k)
    : Matrix_Format(m, n, nnz, k)
{
    int num_threads = omp_get_max_threads();
    double time_balance;

    #ifdef CUSTOM_COO_VEC_PREFETCH
        const char * env_prefetch_distance = getenv("PREFETCH_DIST");
        if (env_prefetch_distance != NULL)
            prefetch_distance = atol(env_prefetch_distance);
    #endif

    row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
    col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
    a = (ValueType *) malloc(nnz * sizeof(*a));

    #pragma omp parallel for schedule(dynamic, 1024)
    for (long i = 0; i < m; i++) {
        for (long j = csr_ia[i]; j < csr_ia[i+1]; j++) {
            row_ind[j] = i;
            col_ind[j] = csr_ja[j];
            #ifdef CUSTOM_COO_VEC_XROW_COLIND0
                col_ind[j] = csr_ja[0];
            #endif
            a[j] = csr_a[j];
        }
    }

    // Group A reorderings: relabel rows/cols (keeps the arrays row-sorted).
    reorder_relabel(this, csr_ia, csr_ja);

    thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
    thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
    thread_row_s = (INT_T *) malloc(num_threads * sizeof(*thread_row_s));
    thread_row_e = (INT_T *) malloc(num_threads * sizeof(*thread_row_e));
    thread_v_e = (ValueType *) malloc(num_threads * k * sizeof(*thread_v_e));

    time_balance = time_it(1,
        _Pragma("omp parallel")
        {
            int tnum = omp_get_thread_num();
            loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

            #ifdef COO_PARTITION_ROW_SPLIT
                if (tnum > 0 && thread_j_s[tnum] < nnz) {
                    while(thread_j_s[tnum] < nnz && row_ind[thread_j_s[tnum]] == row_ind[thread_j_s[tnum] - 1]) {
                        thread_j_s[tnum]++;
                    }
                }

                _Pragma("omp barrier")

                // Set end based on neighbor's start
                if (tnum == num_threads - 1) {
                    thread_j_e[tnum] = nnz;
                } else {
                    thread_j_e[tnum] = thread_j_s[tnum + 1];
                }

                // Safety check: If a single row is massive, a thread might have start > end.
                // We clamp it to ensure loops don't break.
                if (thread_j_e[tnum] < thread_j_s[tnum]) {
                    thread_j_e[tnum] = thread_j_s[tnum];
                }
            #endif

            // The arrays are still row-sorted here, so the first and last non-zero give the row range of the thread.
            if (thread_j_e[tnum] > thread_j_s[tnum]) {
                thread_row_s[tnum] = row_ind[thread_j_s[tnum]];
                thread_row_e[tnum] = row_ind[thread_j_e[tnum] - 1];
            } else {
                thread_row_s[tnum] = 0;
                thread_row_e[tnum] = -1;
            }
        }
    );

    // Group B reorderings: reorder the non-zeros inside each thread segment.
    reorder_segments(this);

    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK) && !COO_FIXEDK_ACCUMULATE
        // STORE mode writes y[row] once per run, so every row must be a single run: row_ind must be non-decreasing inside
        // each segment (rows of different segments are disjoint because of the row split).
        long num_violations = 0;
        #pragma omp parallel reduction(+:num_violations)
        {
            int tnum = omp_get_thread_num();
            for (long j = thread_j_s[tnum] + 1; j < thread_j_e[tnum]; j++)
                if (row_ind[j] < row_ind[j - 1])
                    num_violations++;
        }
        if (num_violations > 0) {
            fprintf(stderr, "kernel_coo.cpp: fixedk STORE mode needs every row to be a single run, but row_ind decreases at %ld positions inside the thread segments. Use the ACCUMULATE mode.\n", num_violations);
            exit(1);
        }
    #endif

    // Base kernel preprocessing on the final order (fixedk ACCUMULATE: first-run flags and empty rows).
    base_preprocess(this);

    // Diagnostic: runs of the final non-zero order (maximal stretches of consecutive non-zeros of the same row inside a
    // thread segment). avg_run_length = nnz / runs; runs_per_row = runs / non-empty rows (1.0 = every row is one run).
    if (getenv("COO_PRINT_RUNS") != NULL && atoi(getenv("COO_PRINT_RUNS"))) {
        long num_runs = 0;
        long num_nonempty_rows = 0;
        #pragma omp parallel reduction(+:num_runs)
        {
            int tnum = omp_get_thread_num();
            for (long j = thread_j_s[tnum]; j < thread_j_e[tnum]; j++)
                if (j == thread_j_s[tnum] || row_ind[j] != row_ind[j - 1])
                    num_runs++;
        }
        #pragma omp parallel for reduction(+:num_nonempty_rows)
        for (long i = 0; i < m; i++)
            if (csr_ia[i + 1] > csr_ia[i])
                num_nonempty_rows++;
        printf("RUNS nnz=%ld runs=%ld avg_run_length=%.4lf nonempty_rows=%ld avg_row_length=%.4lf runs_per_row=%.4lf\n",
                nnz, num_runs, (double) nnz / num_runs, num_nonempty_rows, (double) nnz / num_nonempty_rows,
                (double) num_runs / num_nonempty_rows);
    }

    // Diagnostic: hash of the final non-zero order, to compare with the old reference files (same print there).
    if (getenv("COO_PRINT_ORDER_HASH") != NULL && atoi(getenv("COO_PRINT_ORDER_HASH"))) {
        uint64_t h = 1469598103934665603ULL;   // FNV-1a over (row, col) of every non-zero, in order
        for (long j = 0; j < nnz; j++) {
            h = (h ^ (uint32_t) row_ind[j]) * 1099511628211ULL;
            h = (h ^ (uint32_t) col_ind[j]) * 1099511628211ULL;
        }
        printf("ORDER_HASH %016llx\n", (unsigned long long) h);
    }

    #ifdef PRINT_STATISTICS
        long i;
        num_loops = 0;
        thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
        thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
        for (i=0;i<num_threads;i++)
        {
            printf("Thread %ld: nnz range [%d, %d) nnz: %ld of nnz_total: %ld\n", i, thread_j_s[i], thread_j_e[i], (long) (thread_j_e[i] - thread_j_s[i]), nnz);
        }
    #endif
}

COOArrays::~COOArrays()
{
    free(a);
    free(row_ind);
    free(col_ind);
    free(thread_j_s);
    free(thread_j_e);
    free(thread_row_s);
    free(thread_row_e);
    free(thread_v_e);

    reorder_free(this);
    base_free(this);

    #ifdef PRINT_STATISTICS
        free(thread_time_barrier);
        free(thread_time_compute);
    #endif
}

// Forward declarations
void compute_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
COOArrays::spmm(ValueType * x, ValueType * y, int k)
{
    ValueType * x_k, * y_k;   // The dense buffers the kernel works on (permuted copies, or x / y themselves).

    num_loops++;
    reorder_spmm_begin(this, x, y, k, &x_k, &y_k);
    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
        compute_coo_vector_xrow_row_reg_fixedk(this, x_k, y_k, k);
    #elif defined(CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE)
        compute_coo_vector_xrow_perfect_nnz_balance(this, x_k, y_k, k);
    #else
        compute_coo_vector_xrow(this, x_k, y_k, k);
    #endif
    reorder_spmm_end(this, y_k, y, k);
}

void
COOArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    ValueType * x_k, * y_k, * out_k;

    reorder_sddmm_begin(this, x, y, out, k, &x_k, &y_k, &out_k);
    compute_coo_sddmm(this, x_k, y_k, out_k, k);
    reorder_sddmm_end(this, out_k, out);
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COOArrays * coo = new COOArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK)
        #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_ACCUMULATE)
            #define COO_BASE_NAME "COO_RowSplit_Vec_RowReg_FixedK_Acc"
        #else
            #define COO_BASE_NAME "COO_RowSplit_Vec_RowReg_FixedK"
        #endif
    #elif defined(CUSTOM_COO_VEC_XROW_COLIND0)
        #define COO_BASE_NAME "COO_ColInd0_Vec"
    #elif defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT)
        #define COO_BASE_NAME "COO_RowSplit_Vec"
    #elif defined(CUSTOM_COO_VEC_XROW_ATOMIC)
        #define COO_BASE_NAME "COO_Atomic_Vec"
    #elif defined(CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE)
        #define COO_BASE_NAME "COO_Vec_PBV"
    #endif
    #if defined(CUSTOM_COO_VEC_PREFETCH_FULL_ROW)
        #define COO_PREFETCH_NAME "_PrefetchRow"
    #elif defined(CUSTOM_COO_VEC_PREFETCH)
        #define COO_PREFETCH_NAME "_Prefetch"
    #else
        #define COO_PREFETCH_NAME ""
    #endif
    #if defined(CUSTOM_COO_VEC_ZERO_STATIC)
        #define COO_ZERO_NAME "_ZeroStatic"
    #elif defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_ZERO_ALL)
        #define COO_ZERO_NAME "_ZeroAll"
    #else
        #define COO_ZERO_NAME ""
    #endif
    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_RUN_SCAN)
        #define COO_RUN_NAME "_RunScan"
    #else
        #define COO_RUN_NAME ""
    #endif
    #define COO_XSTR(s) COO_STR(s)
    #define COO_STR(s) #s
    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE)
        #define COO_SLICE_NAME "_Slice" COO_XSTR(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE)
    #else
        #define COO_SLICE_NAME ""
    #endif
    #if defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER_PACK)
        #define COO_OUTER_NAME "_Outer" COO_XSTR(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER) "_Pack"
    #elif defined(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER)
        #define COO_OUTER_NAME "_Outer" COO_XSTR(CUSTOM_COO_VEC_ROW_REG_FIXEDK_SLICE_OUTER)
    #else
        #define COO_OUTER_NAME ""
    #endif
    coo->format_name = (char *) COO_BASE_NAME COO_SLICE_NAME COO_OUTER_NAME COO_RUN_NAME COO_PREFETCH_NAME COO_ZERO_NAME COO_REORDER_NAME;
    return coo;
}

//==========================================================================================================================================
//= SDDMM
//==========================================================================================================================================

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
