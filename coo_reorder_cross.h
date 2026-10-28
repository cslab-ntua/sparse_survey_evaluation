#ifndef COO_REORDER_CROSS_H
#define COO_REORDER_CROSS_H

#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_map>

//==========================================================================================================================================
//= Reordering: greedy "cross" traversal (group B, COO_REORDER_CROSS)
//==========================================================================================================================================
//
// Greedy walk over the non-zeros of each thread segment. After a non-zero (r, c), the next one is picked from its "cross":
// the unvisited non-zeros of row r and of column c (at most CROSS_DISTANCE_LIMIT of each). A simulated cache remembers when
// every row / column was last used: a candidate whose row and column are both still "in cache" (used within the last
// CACHE_CAPACITY steps) is taken immediately, otherwise the candidate with the best score (cache hits + row / column
// populations) wins. If the cross is empty, the walk restarts from the SEED_ROW most populated rows of the segment, and
// finally from the first unvisited non-zero.
//
// Parameters (env, default = run.sh value): SEED_ROW (50), CROSS_DISTANCE_LIMIT (100), CACHE_CAPACITY (256).
//
// Rows are split into several runs (COO_REORDER_KEEPS_ROWS = 0, fixedk uses ACCUMULATE mode).
//
// Ported unchanged from kernel_coo_vec_cross.cpp (preprocess_cross). Note: the candidate gathering walks the whole row /
// column list of the last non-zero at every step, so segments with very dense rows or columns take long to preprocess.

#define COO_REORDER_KEEPS_ROWS  0
#define COO_REORDER_NAME        "_Cross"

#include "coo_reorder_segment_common.h"

static inline
void
reorder_segments(struct COOArrays * coo)
{
    const int r_first_rows = segment_env_long("SEED_ROW", 50);
    const int cross_distance_limit = segment_env_long("CROSS_DISTANCE_LIMIT", 100);
    const int cache_capacity = segment_env_long("CACHE_CAPACITY", 256);

    INT_T * row_ind = coo->row_ind;
    INT_T * col_ind = coo->col_ind;

    segment_order_init(coo);

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];
        long local_nnz = j_e - j_s;

        if (local_nnz > 0) {
            // 1. Build local adjacency lists (The "Cross" lookup)
            std::unordered_map<long, std::vector<long>> row_to_nnzs;
            std::unordered_map<long, std::vector<long>> col_to_nnzs;
            std::unordered_map<long, int> row_pop_count;

            for (long j = j_s; j < j_e; j++) {
                row_to_nnzs[row_ind[j]].push_back(j);
                col_to_nnzs[col_ind[j]].push_back(j);
                row_pop_count[row_ind[j]]++;
            }

            // 2. Find the 'r_first_rows' most populated rows (Seeds)
            std::vector<std::pair<long, int>> pop_rows(row_pop_count.begin(), row_pop_count.end());
            std::sort(pop_rows.begin(), pop_rows.end(),
                [](const std::pair<long, int>& a, const std::pair<long, int>& b) {
                    return a.second > b.second;
                });

            std::vector<long> seeds;
            int seed_limit = std::min((int)pop_rows.size(), r_first_rows);
            for (int i = 0; i < seed_limit; i++) {
                seeds.push_back(pop_rows[i].first);
            }

            // 3. Setup state tracking
            std::vector<bool> visited(local_nnz, false);
            std::unordered_map<long, long> row_last_seen;
            std::unordered_map<long, long> col_last_seen;
            std::vector<long> reordered_j;
            reordered_j.reserve(local_nnz);

            long current_step = 0;

            // Helper lambda to check if an index is in our simulated cache
            auto in_cache = [&](long r, long c) {
                bool r_hit = (row_last_seen.count(r) && (current_step - row_last_seen[r] <= cache_capacity));
                bool c_hit = (col_last_seen.count(c) && (current_step - col_last_seen[c] <= cache_capacity));
                return std::make_pair(r_hit, c_hit);
            };

            // 4. Main Greedy Loop
            long last_processed_j = -1;

            while ((long) reordered_j.size() < local_nnz) {
                long best_candidate = -1;
                long best_score = -1;

                // If we have a last processed element, search its cross
                if (last_processed_j != -1) {
                    long current_r = row_ind[last_processed_j];
                    long current_c = col_ind[last_processed_j];

                    // Gather candidates from the Cross (bounded by cross_distance_limit)
                    std::vector<long> candidates;
                    int count = 0;
                    for (long neighbor_j : row_to_nnzs[current_r]) {
                        if (!visited[neighbor_j - j_s] && count++ < cross_distance_limit)
                            candidates.push_back(neighbor_j);
                    }
                    count = 0;
                    for (long neighbor_j : col_to_nnzs[current_c]) {
                        if (!visited[neighbor_j - j_s] && count++ < cross_distance_limit)
                            candidates.push_back(neighbor_j);
                    }

                    for (long cand_j : candidates) {
                        long cand_r = row_ind[cand_j];
                        long cand_c = col_ind[cand_j];

                        auto cache_status = in_cache(cand_r, cand_c);

                        // Condition 1: Both row and col are already in cache
                        if (cache_status.first && cache_status.second) {
                            best_candidate = cand_j;
                            break;
                        }

                        // Condition 2: Score by intersection of candidate's cross with cache
                        long score = 0;
                        if (cache_status.first) score += 10; // Weight row hit
                        if (cache_status.second) score += 10; // Weight col hit

                        // Add secondary score based on neighbors in cache (simplified)
                        score += row_to_nnzs[cand_r].size() + col_to_nnzs[cand_c].size();

                        if (score > best_score) {
                            best_score = score;
                            best_candidate = cand_j;
                        }
                    }
                }

                // Fallback: If cross search failed or we are just starting, use seeds or linear scan
                if (best_candidate == -1) {
                    bool found = false;
                    // Try seeds first
                    for (long seed_r : seeds) {
                        for (long cand_j : row_to_nnzs[seed_r]) {
                            if (!visited[cand_j - j_s]) {
                                best_candidate = cand_j;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }

                    // Absolute fallback: first unvisited
                    if (!found) {
                        for (long j = j_s; j < j_e; j++) {
                            if (!visited[j - j_s]) {
                                best_candidate = j;
                                break;
                            }
                        }
                    }
                }

                // Commit the selected element
                visited[best_candidate - j_s] = true;
                reordered_j.push_back(best_candidate);

                // Update cache state
                row_last_seen[row_ind[best_candidate]] = current_step;
                col_last_seen[col_ind[best_candidate]] = current_step;

                last_processed_j = best_candidate;
                current_step++;
            }

            // 5. Apply the new ordering
            segment_apply_order(coo, j_s, local_nnz, reordered_j.data());
        }
    }
}

#endif /* COO_REORDER_CROSS_H */
