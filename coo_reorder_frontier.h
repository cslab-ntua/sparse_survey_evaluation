#ifndef COO_REORDER_FRONTIER_H
#define COO_REORDER_FRONTIER_H

#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_map>

//==========================================================================================================================================
//= Reordering: greedy frontier traversal (group B, COO_REORDER_FRONTIER)
//==========================================================================================================================================
//
// Greedy walk over the non-zeros of each thread segment, driven by a frontier of candidates: every chosen non-zero (r, c)
// adds the unvisited non-zeros of row r and column c to the frontier. At every step all frontier candidates are scored with
//
//     score(v) = ALPHA * R(v) + BETA * N(v)
//
//   R(v)  recency: +1 for the row and +1 for the column of v if they were used within the last CACHE_CAPACITY steps
//   N(v)  coverage: fraction of the row's + fraction of the column's non-zeros already visited
//
// the best one is taken, and the frontier is clamped to the FRONTIER_SIZE best. If the frontier is empty, the walk restarts
// from the SEED_ROW most populated rows of the segment, and finally from the first unvisited non-zero.
//
// Parameters (env, default = run.sh value): SEED_ROW (50), FRONTIER_SIZE (10000), CACHE_CAPACITY (256), ALPHA (1.0),
// BETA (1.0).
//
// Rows are split into several runs (COO_REORDER_KEEPS_ROWS = 0, fixedk uses ACCUMULATE mode).
//
// Ported unchanged from kernel_coo_vec_frontier.cpp (preprocess_frontier). Note: the whole frontier (up to FRONTIER_SIZE
// candidates) is scored and sorted at every step, so preprocessing costs about nnz * FRONTIER_SIZE * log(FRONTIER_SIZE) and
// takes long on large matrices.

#define COO_REORDER_KEEPS_ROWS  0
#define COO_REORDER_NAME        "_Frontier"

#include "coo_reorder_segment_common.h"

static inline
void
reorder_segments(struct COOArrays * coo)
{
    const int r_first_rows = segment_env_long("SEED_ROW", 50);
    const int frontier_size = segment_env_long("FRONTIER_SIZE", 10000);
    const int cache_capacity = segment_env_long("CACHE_CAPACITY", 256);
    const float alpha = segment_env_double("ALPHA", 1.0);
    const float beta = segment_env_double("BETA", 1.0);

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
            // 1. Build local adjacency lists and degree counters
            std::unordered_map<long, std::vector<long>> row_nnzs_set;
            std::unordered_map<long, std::vector<long>> col_nnzs_set;
            std::unordered_map<long, int> row_count;
            std::unordered_map<long, int> col_count;

            for (long j = j_s; j < j_e; j++) {
                row_nnzs_set[row_ind[j]].push_back(j);
                col_nnzs_set[col_ind[j]].push_back(j);
                row_count[row_ind[j]]++;
                col_count[col_ind[j]]++;
            }

            // 2. Find the 'r_first_rows' most populated rows (Seeds)
            std::vector<std::pair<long, int>> row_count_vector(row_count.begin(), row_count.end());
            std::sort(row_count_vector.begin(), row_count_vector.end(),
                [](const std::pair<long, int>& a, const std::pair<long, int>& b) {
                    return a.second > b.second;
                });

            std::vector<long> seeds;
            int seed_limit = std::min((int)row_count_vector.size(), r_first_rows);
            for (int i = 0; i < seed_limit; i++) {
                seeds.push_back(row_count_vector[i].first);
            }

            // 3. Setup Heuristic State Tracking (The Dictionaries)
            std::vector<bool> visited(local_nnz, false);
            std::vector<bool> in_frontier(local_nnz, false); // Prevents duplicate entries

            std::unordered_map<long, long> row_freshness;     // For R(v) Recency
            std::unordered_map<long, long> col_freshness;     // For R(v) Recency
            std::unordered_map<long, int> row_visited_count;  // For N(v) Coverage
            std::unordered_map<long, int> col_visited_count;  // For N(v) Coverage

            std::vector<long> reordered_j;
            reordered_j.reserve(local_nnz);

            std::vector<long> frontier;
            // Bounding the frontier size to represent our L1/L2 capacity proxy
            size_t MAX_FRONTIER_SIZE = (size_t)frontier_size;

            // Heuristic Weights
            const double ALPHA = alpha; // Weight for Recency R(v)
            const double BETA  = beta; // Weight for Coverage N(v)

            long current_step = 1; // Start at 1 so uninitialized maps (0) are naturally "old"

            // Lambda to compute: score(v) = a*R(v) + b*N(v)
            auto compute_score = [&](long j) -> double {
                long r = row_ind[j];
                long c = col_ind[j];

                // R(v): Recency (binary: row / col seen within the last cache_capacity steps)
                double r_score = 0.0;
                long r_age = current_step - row_freshness[r];
                long c_age = current_step - col_freshness[c];

                if (row_freshness[r] > 0 && r_age <= cache_capacity) {
                    r_score += 1.0;
                }
                if (col_freshness[c] > 0 && c_age <= cache_capacity) {
                    r_score += 1.0;
                }

                // N(v): Neighborhood coverage (Fraction of row/col already visited)
                double n_score = 0.0;
                if (!row_nnzs_set[r].empty()) {
                    n_score += (double)row_visited_count[r] / row_count[r];
                }
                if (!col_nnzs_set[c].empty()) {
                    n_score += (double)col_visited_count[c] / col_count[c];
                }

                return (ALPHA * r_score) + (BETA * n_score);
            };

            // 4. Main Greedy Heuristic Loop
            while ((long) reordered_j.size() < local_nnz) {
                long best_candidate = -1;

                // --- A & E COMBINED: Score, Sort, Pick, and Clamp ---
                if (!frontier.empty()) {
                    std::vector<std::pair<double, long>> scored_frontier;
                    scored_frontier.reserve(frontier.size());

                    // 1. Score everything once
                    for (long cand : frontier) {
                        scored_frontier.push_back({compute_score(cand), cand});
                    }

                    // 2. Sort descending by score
                    std::sort(scored_frontier.begin(), scored_frontier.end(),
                        [](const std::pair<double, long>& a, const std::pair<double, long>& b) {
                            return a.first > b.first;
                        });

                    // 3. The absolute best candidate is right at the top
                    best_candidate = scored_frontier[0].second;
                    in_frontier[best_candidate - j_s] = false;

                    // 4. Rebuild the frontier, applying the MAX_FRONTIER_SIZE clamp natively
                    frontier.clear();

                    // We start at i = 1 because i = 0 is our best_candidate.
                    // We keep up to MAX_FRONTIER_SIZE elements.
                    size_t keep_count = std::min(scored_frontier.size(), MAX_FRONTIER_SIZE + 1);

                    for (size_t i = 1; i < keep_count; i++) {
                        frontier.push_back(scored_frontier[i].second);
                    }

                    // 5. Un-flag the items that were evicted
                    for (size_t i = keep_count; i < scored_frontier.size(); i++) {
                        in_frontier[scored_frontier[i].second - j_s] = false;
                    }
                }

                // --- B. Fallback if Frontier is empty ---
                if (best_candidate == -1) {
                    bool found = false;
                    // Try seeds first
                    for (long seed_r : seeds) {
                        for (long cand_j : row_nnzs_set[seed_r]) {
                            if (!visited[cand_j - j_s]) {
                                best_candidate = cand_j;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    // Absolute fallback: linear scan
                    if (!found) {
                        for (long j = j_s; j < j_e; j++) {
                            if (!visited[j - j_s]) {
                                best_candidate = j;
                                break;
                            }
                        }
                    }
                }

                // --- C. Commit the Element ---
                visited[best_candidate - j_s] = true;
                reordered_j.push_back(best_candidate);
                long best_r = row_ind[best_candidate];
                long best_c = col_ind[best_candidate];

                // Update dictionaries
                row_freshness[best_r] = current_step;
                col_freshness[best_c] = current_step;
                row_visited_count[best_r]++;
                col_visited_count[best_c]++;

                // --- D. Expand Frontier with Neighbors ---
                // (We just throw them in; they will be scored and sorted at the start of the next loop!)

                // Add Row Neighbors
                for (long neighbor_j : row_nnzs_set[best_r]) {
                    long local_idx = neighbor_j - j_s;
                    if (!visited[local_idx] && !in_frontier[local_idx]) {
                        frontier.push_back(neighbor_j);
                        in_frontier[local_idx] = true;
                    }
                }

                // Add Col Neighbors
                for (long neighbor_j : col_nnzs_set[best_c]) {
                    long local_idx = neighbor_j - j_s;
                    if (!visited[local_idx] && !in_frontier[local_idx]) {
                        frontier.push_back(neighbor_j);
                        in_frontier[local_idx] = true;
                    }
                }

                current_step++;
            }

            // 5. Apply the new ordering
            segment_apply_order(coo, j_s, local_nnz, reordered_j.data());
        }
    }
}

#endif /* COO_REORDER_FRONTIER_H */
