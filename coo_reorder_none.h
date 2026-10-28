#ifndef COO_REORDER_NONE_H
#define COO_REORDER_NONE_H

//==========================================================================================================================================
//= Reordering: none
//==========================================================================================================================================
//
// Every coo_reorder_*.h provides the same interface, so that kernel_coo.cpp has a single call site for each step:
//
//   COO_REORDER_KEEPS_ROWS   1 if, after the reordering, the non-zeros of each row are still contiguous inside the thread's
//                            segment (row-sorted order, e.g. none / RCM), 0 if rows can be split into several runs
//                            (traversal orders, e.g. z-order / hilbert / frontier). The fixedk base uses this to pick
//                            STORE (y = acc) or ACCUMULATE (y += acc) mode.
//
//   COO_REORDER_NAME         Suffix appended to the format name.
//
//   reorder_relabel(coo, csr_ia, csr_ja)
//                            Group A: global relabeling of rows/cols. Called after the COO arrays are built, before the
//                            partitioning. Must leave the COO arrays row-sorted.
//
//   reorder_segments(coo)    Group B: reordering of the non-zeros inside each thread segment [thread_j_s, thread_j_e).
//                            Called after the partitioning. The row range of each thread (thread_row_s, thread_row_e) is
//                            computed before this call, so the kernels don't depend on the order inside the segment.
//
//   reorder_spmm_begin(coo, x, y, k, &x_k, &y_k)
//   reorder_spmm_end(coo, y_k, y, k)
//                            Called around the SpMM kernel (inside the timed region). begin gives the dense buffers the kernel
//                            works on (x_k, y_k), end writes the result back to y in the original order if needed.
//
//   reorder_sddmm_begin(coo, x, y, out, k, &x_k, &y_k, &out_k)
//   reorder_sddmm_end(coo, out_k, out)
//                            Same for SDDMM (out is in the original CSR non-zero order).
//
//   reorder_free(coo)        Frees whatever the reordering allocated. Called from the destructor.

#define COO_REORDER_KEEPS_ROWS  1
#define COO_REORDER_NAME        ""

static inline
void
reorder_relabel(__attribute__((unused)) struct COOArrays * coo, __attribute__((unused)) INT_T * csr_ia, __attribute__((unused)) INT_T * csr_ja)
{
}

static inline
void
reorder_segments(__attribute__((unused)) struct COOArrays * coo)
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
reorder_sddmm_begin(__attribute__((unused)) struct COOArrays * coo, ValueType * x, ValueType * y, ValueType * out, __attribute__((unused)) int k, ValueType ** x_k, ValueType ** y_k, ValueType ** out_k)
{
    *x_k = x;
    *y_k = y;
    *out_k = out;
}

static inline
void
reorder_sddmm_end(__attribute__((unused)) struct COOArrays * coo, __attribute__((unused)) ValueType * out_k, __attribute__((unused)) ValueType * out)
{
}

static inline
void
reorder_free(__attribute__((unused)) struct COOArrays * coo)
{
}

#endif /* COO_REORDER_NONE_H */
