#ifndef COO_REORDER_SEGMENT_COMMON_H
#define COO_REORDER_SEGMENT_COMMON_H

#include <vector>

//==========================================================================================================================================
//= Common part of the group B reorderings (traversal orders inside each thread segment)
//==========================================================================================================================================
//
// A group B reordering header defines COO_REORDER_KEEPS_ROWS, COO_REORDER_NAME and reorder_segments(), includes this file,
// and in reorder_segments() calls:
//
//   segment_order_init(coo)                         once, before any reordering
//   segment_apply_order(coo, js, size, order)       once per thread segment (from that thread), with the new order of the
//                                                   segment as global non-zero indices
//
// Everything else of the interface is provided here: no relabeling, the dense operands are not touched (rows and columns keep
// their ids, only the order of the non-zeros changes), and SDDMM writes its output in the reordered non-zero order and then
// scatters it back to the original CSR order through seg_nnz_perm.

long * seg_nnz_perm = NULL;     // original CSR position of the j-th non-zero after the reordering
ValueType * seg_out = NULL;     // SDDMM output in the reordered non-zero order, allocated on the first sddmm() call

// Parameters of the heuristic reorderings (cross, frontier, ring-buffer) come from env variables, as in run.sh.
// A missing variable falls back to the given default (the run.sh value) instead of crashing.
static inline
long
segment_env_long(const char * name, long default_value)
{
    const char * str = getenv(name);
    return (str != NULL) ? atol(str) : default_value;
}

static inline
double
segment_env_double(const char * name, double default_value)
{
    const char * str = getenv(name);
    return (str != NULL) ? atof(str) : default_value;
}

static inline
void
segment_order_init(struct COOArrays * coo)
{
    seg_nnz_perm = (long *) malloc(coo->nnz * sizeof(*seg_nnz_perm));
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < coo->nnz; j++)
        seg_nnz_perm[j] = j;
}

// Reorder the non-zeros of the segment [js, js + size): the i-th non-zero of the segment becomes the one that was at global
// index order[i]. order must be a permutation of [js, js + size).
static inline
void
segment_apply_order(struct COOArrays * coo, long js, long size, const long * order)
{
    std::vector<INT_T> t_row(size);
    std::vector<INT_T> t_col(size);
    std::vector<ValueType> t_val(size);
    std::vector<long> t_perm(size);

    for (long i = 0; i < size; i++) {
        long src_idx = order[i];
        t_row[i] = coo->row_ind[src_idx];
        t_col[i] = coo->col_ind[src_idx];
        t_val[i] = coo->a[src_idx];
        t_perm[i] = seg_nnz_perm[src_idx];
    }

    for (long i = 0; i < size; i++) {
        long dst_idx = js + i;
        coo->row_ind[dst_idx] = t_row[i];
        coo->col_ind[dst_idx] = t_col[i];
        coo->a[dst_idx] = t_val[i];
        seg_nnz_perm[dst_idx] = t_perm[i];
    }
}

// ==========================================================================================================================================
// = Reordering interface (everything except reorder_segments)
// ==========================================================================================================================================

static inline
void
reorder_relabel(__attribute__((unused)) struct COOArrays * coo, __attribute__((unused)) INT_T * csr_ia, __attribute__((unused)) INT_T * csr_ja)
{
}

static inline
void
reorder_spmm_begin(__attribute__((unused)) struct COOArrays * coo, ValueType * x, ValueType * y, __attribute__((unused)) int k, ValueType ** x_k, ValueType ** y_k)
{
    *x_k = x;
    *y_k = y;
}

static inline
void
reorder_spmm_end(__attribute__((unused)) struct COOArrays * coo, __attribute__((unused)) ValueType * y_k, __attribute__((unused)) ValueType * y, __attribute__((unused)) int k)
{
}

static inline
void
reorder_sddmm_begin(struct COOArrays * coo, ValueType * x, ValueType * y, __attribute__((unused)) ValueType * out, __attribute__((unused)) int k, ValueType ** x_k, ValueType ** y_k, ValueType ** out_k)
{
    if (seg_out == NULL)
        seg_out = (ValueType *) malloc(coo->nnz * sizeof(*seg_out));
    *x_k = x;
    *y_k = y;
    *out_k = seg_out;
}

static inline
void
reorder_sddmm_end(struct COOArrays * coo, ValueType * out_k, ValueType * out)
{
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < coo->nnz; j++)
        out[seg_nnz_perm[j]] = out_k[j];
}

static inline
void
reorder_free(__attribute__((unused)) struct COOArrays * coo)
{
    free(seg_nnz_perm);
    free(seg_out);
}

#endif /* COO_REORDER_SEGMENT_COMMON_H */
