#ifndef COO_REORDER_HILBERT_H
#define COO_REORDER_HILBERT_H

#include <vector>
#include <algorithm>
#include <utility>
#include <stdint.h>

//==========================================================================================================================================
//= Reordering: segmented Hilbert curve (group B, COO_REORDER_HILBERT)
//==========================================================================================================================================
//
// The non-zeros of each thread segment are sorted by the distance of (row, col) along a Hilbert curve. Like Z-order it
// keeps nearby (row, col) pairs close in the traversal, but without Z-order's long jumps between quadrants. The segments
// themselves are not changed, so every row stays owned by a single thread, but a row is split into several runs inside its
// segment (COO_REORDER_KEEPS_ROWS = 0, fixedk uses ACCUMULATE mode).

#define COO_REORDER_KEEPS_ROWS  0
#define COO_REORDER_NAME        "_Hilbert"

#include "coo_reorder_segment_common.h"

// Converts 2D coordinates (row, col) into a 1D Hilbert Curve distance
static inline uint64_t hilbert_c2d(uint32_t x, uint32_t y) {
    uint64_t d = 0;
    // Process 32 bits from MSB to LSB
    for (int s = 31; s >= 0; s--) {
        uint32_t rx = (x >> s) & 1;
        uint32_t ry = (y >> s) & 1;

        // Append quadrant bits to distance
        d = (d << 2) | ((3 * rx) ^ ry);

        // Rotate and flip coordinates for the next iteration if necessary
        if (ry == 0) {
            if (rx == 1) {
                x = ~x;
                y = ~y;
            }
            // Swap x and y
            uint32_t t = x;
            x = y;
            y = t;
        }
    }
    return d;
}

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
            // (Hilbert distance, global index) pairs: sorting them gives the Hilbert order, with ties (duplicate entries)
            // kept in the original order.
            std::vector<std::pair<uint64_t, long>> keys(size);
            for (long i = 0; i < size; i++) {
                long j = js + i;
                keys[i] = { hilbert_c2d((uint32_t) coo->row_ind[j], (uint32_t) coo->col_ind[j]), j };
            }
            std::sort(keys.begin(), keys.end());

            std::vector<long> order(size);
            for (long i = 0; i < size; i++)
                order[i] = keys[i].second;

            segment_apply_order(coo, js, size, order.data());
        }
    }
}

#endif /* COO_REORDER_HILBERT_H */
