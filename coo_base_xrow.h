#ifndef COO_BASE_XROW_H
#define COO_BASE_XROW_H

//==========================================================================================================================================
//= Base kernels: per-nnz vectorized COO (row_split / atomic / perfect nnz balance)
//==========================================================================================================================================
//
// The original kernel_coo_vec.cpp kernels. Every non-zero does y[row] += val * x[col] directly on y, so they are correct for
// any order of the non-zeros inside a thread segment (they don't care about runs).
//
// The row ranges used for the zero-initialization (thread_row_s, thread_row_e) are computed at partitioning time, before any
// reordering of the segments, so they stay valid after a traversal reordering (row_ind[j_s] / row_ind[j_e-1] would not be).

//==========================================================================================================================================
//= Preprocessing
//==========================================================================================================================================

// Nothing to precompute for the per-nnz kernels (base interface, see coo_base_row_reg_fixedk.h).
static inline
void
base_preprocess(__attribute__((unused)) COOArrays * coo)
{
}

static inline
void
base_free(__attribute__((unused)) COOArrays * coo)
{
}

//==========================================================================================================================================
//= Subkernels COO
//==========================================================================================================================================

__attribute__((hot))
static inline
void
subkernel_val_coo_vec_xrow_atomic(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
{
    long r = coo->row_ind[j];
    long c_idx = coo->col_ind[j];
    ValueType val = coo->a[j];

    long c, c_e_vector;
    const long mask = ~(((long) VEC_LEN) - 1);

    vec_t(VTF, VEC_LEN) v_val, v_x, v_prod;

    c_e_vector = k & mask;
    v_val = vec_set1(VTF, VEC_LEN, val);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        v_x   = vec_loadu(VTF, VEC_LEN, &x[c_idx * k + c]);
        v_prod = vec_mul(VTF, VEC_LEN, v_val, v_x);

        ValueType temp[VEC_LEN];
        vec_storeu(VTF, VEC_LEN, temp, v_prod);
        for(int v = 0; v < VEC_LEN; ++v) {
             #pragma omp atomic
             y[r * k + c + v] += temp[v];
        }
    }

    for (c = c_e_vector; c < k; c++) {
        ValueType product = val * x[c_idx * k + c];
        #pragma omp atomic
        y[r * k + c] += product;
    }
}

__attribute__((hot))
static inline
void
subkernel_val_coo_vec_xrow_noatomic(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
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

//==========================================================================================================================================
//= COO Main Computation Kernels
//==========================================================================================================================================

// Used by CUSTOM_COO_VEC_XROW_ROW_SPLIT and CUSTOM_COO_VEC_XROW_ATOMIC.
void
compute_coo_vector_xrow(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s, j_e, start_row, end_row;
        j_s = thread_j_s[tnum];
        j_e = thread_j_e[tnum];
        // Empty threads (e.g. nnz < num_threads) have start_row = 0, end_row = -1, so the zero-init below is a no-op.
        start_row = thread_row_s[tnum];
        end_row = thread_row_e[tnum];
        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        #ifdef CUSTOM_COO_VEC_ZERO_STATIC
            // One static (row-balanced) loop over all of y, instead of the nnz-balanced row ranges. Its implicit barrier
            // also makes every thread wait for the zeroing before adding to rows of other threads (atomic).
            (void) start_row;
            (void) end_row;
            _Pragma("omp for schedule(static)")
            for (long i = 0; i < coo->m * k; i++)
                y[i] = 0;
        #else
            for (long i=start_row;i<=end_row;i++)
            {
                for (long c = 0; c < k; c++)
                    y[i*k + c]=0;
            }

            #ifdef CUSTOM_COO_VEC_XROW_ATOMIC
                // Without row splitting a boundary row is shared by two threads: every thread must finish zeroing before any
                // thread starts adding to it.
                _Pragma("omp barrier")
            #endif
        #endif

        for (long j = j_s; j < j_e; j++)
        {
            #ifdef CUSTOM_COO_VEC_PREFETCH
                // Prefetch the y and x rows of a future non-zero of this segment into the cache
                // (the first cache line of each row, as in the old kernel_coo_z_order.cpp, or all of them with
                // CUSTOM_COO_VEC_PREFETCH_FULL_ROW).
                if (j + prefetch_distance < j_e) {
                    long future_r = coo->row_ind[j + prefetch_distance];
                    long future_c = coo->col_ind[j + prefetch_distance];
                    COO_PREFETCH_ROW(&y[future_r * k], k, 1);   // y: for writing (1)
                    COO_PREFETCH_ROW(&x[future_c * k], k, 0);   // x: for reading (0)
                }
            #endif
            #if defined(CUSTOM_COO_VEC_XROW_ATOMIC)
                subkernel_val_coo_vec_xrow_atomic(coo, x, y, j, k);
            #else
                subkernel_val_coo_vec_xrow_noatomic(coo, x, y, j, k);
            #endif
        }

        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1, _Pragma("omp barrier"));
        thread_time_barrier[tnum] += time;
        #endif
    }
}

// Used by CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE.
void
compute_coo_vector_xrow_perfect_nnz_balance(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    int num_threads = omp_get_max_threads();

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];

        long boundary_row = -1;
        long start_row = -1;
        long next_start_row = -1; // For the next thread, to know where the next boundary is
        if (j_e > j_s) {
            boundary_row = thread_row_e[tnum];
            start_row = thread_row_s[tnum];
            next_start_row = (tnum < num_threads - 1 && thread_j_e[tnum + 1] > thread_j_s[tnum + 1]) ? thread_row_s[tnum + 1] : coo->m;
        }

        for(long c = 0; c < k; c++) {
            thread_v_e[tnum * k + c] = 0.0;
        }
        #ifdef CUSTOM_COO_VEC_ZERO_STATIC
            // One static (row-balanced) loop over all of y. Its implicit barrier orders the zeroing of a row shared with
            // the previous thread before this thread adds to it.
            (void) start_row;
            (void) next_start_row;
            _Pragma("omp for schedule(static)")
            for (long i = 0; i < coo->m * k; i++)
                y[i] = 0;
        #else
            for (long i=start_row; i < boundary_row; i++) {
                for (long c = 0; c < k; c++) {
                    y[i * k + c] = 0.0;
                }
            }
            if (boundary_row != next_start_row) {
                for (long c = 0; c < k; c++) {
                    y[boundary_row * k + c] = 0.0;
                }
            }
        #endif

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        for (long j = j_s; j < j_e; j++)
        {
            long current_row = coo->row_ind[j];

            if (current_row == boundary_row) {
                subkernel_val_coo_vec_xrow_partial(coo, x, &thread_v_e[tnum * k], j, k);
            } else {
                subkernel_val_coo_vec_xrow_noatomic(coo, x, y, j, k);
            }
        }

        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1, _Pragma("omp barrier"));
        thread_time_barrier[tnum] += time;
        #endif
    }

    for (int t = 0; t < num_threads; t++)
    {
        long j_e_t = thread_j_e[t];
        long j_s_t = thread_j_s[t];

        if (j_e_t > j_s_t) {
            long boundary_row = thread_row_e[t];

            for (long c = 0; c < k; c++) {
                y[boundary_row * k + c] += thread_v_e[t * k + c];
            }
        }
    }
}

#endif /* COO_BASE_XROW_H */
