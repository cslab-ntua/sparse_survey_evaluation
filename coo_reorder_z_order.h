#ifndef COO_REORDER_Z_ORDER_H
#define COO_REORDER_Z_ORDER_H

#include <vector>
#include <algorithm>
#include <utility>
#include <stdint.h>

//==========================================================================================================================================
//= Reordering: segmented Z-order (group B, COO_REORDER_Z_ORDER)
//==========================================================================================================================================
//
// The non-zeros of each thread segment are sorted by the Morton (Z-order) code of (row, col), i.e. the bits of row and col
// interleaved. The segments themselves are not changed, so every row stays owned by a single thread, but a row is split into
// several runs inside its segment (COO_REORDER_KEEPS_ROWS = 0, fixedk uses ACCUMULATE mode).

#define COO_REORDER_KEEPS_ROWS  0
#define COO_REORDER_NAME        "_ZOrder"

#include "coo_reorder_segment_common.h"

static inline
void
reorder_segments(struct COOArrays * coo)
{
    segment_order_init(coo);

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long js = thread_j_s[tnum];
        long je = thread_j_e[tnum];
        long size = je - js;

        if (size > 0) {
            // (Z-value, global index) pairs: sorting them gives the Z-order, with ties (duplicate entries) kept in the
            // original order.
            std::vector<std::pair<uint64_t, long>> keys(size);
            for (long i = 0; i < size; i++) {
                long j = js + i;
                keys[i] = { bits_u32_interleave((uint32_t) coo->row_ind[j], (uint32_t) coo->col_ind[j]), j };
            }
            std::sort(keys.begin(), keys.end());

            std::vector<long> order(size);
            for (long i = 0; i < size; i++)
                order[i] = keys[i].second;

            segment_apply_order(coo, js, size, order.data());
        }
    }
}

#endif /* COO_REORDER_Z_ORDER_H */
