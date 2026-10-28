#ifndef COO_REORDER_RINGBUFFER_H
#define COO_REORDER_RINGBUFFER_H

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_map>

//==========================================================================================================================================
//= Reordering: greedy frontier traversal with a ring-buffer cache model (group B)
//==========================================================================================================================================
//
// Like the frontier reordering (coo_reorder_frontier.h), every chosen non-zero (r, c) adds the unvisited non-zeros of row r
// and column c to a frontier, and the next non-zero is picked from the frontier. The "cache" is modelled with two ring
// buffers holding the last CACHE_CAPACITY rows and columns used (a row / column is "in cache" while it is in its buffer).
// Three variants, ported unchanged from the old files:
//
//   COO_REORDER_RINGBUFFER            (kernel_coo_vec_ring_buffer.cpp)
//       Every step scores the whole frontier with  ALPHA * (row in cache + col in cache) + BETA * (row + col coverage),
//       takes the best and keeps the FRONTIER_SIZE best. Empty frontier: restart from the SEED_ROW most populated rows,
//       then from the first unvisited non-zero.
//
//   COO_REORDER_RINGBUFFER_ALL_SEED   (kernel_coo_vec_ring_buffer_all_seed.cpp)
//       Same scoring, but the frontier starts pre-filled with all the non-zeros of the SEED_ROW most populated rows and of
//       the SEED_ROW most populated columns. Empty frontier: first unvisited non-zero.
//
//   COO_REORDER_RINGBUFFER_ROULETTE   (kernel_coo_vec_ring_buffer_roulette.cpp)
//       Stochastic: every step draws a number in [0, 100) (rand_r, fixed per-thread seed, so the order is reproducible):
//         < 70  take the frontier candidate of the best tier (row and col in cache > row > col > none), ties by coverage
//         < 90  restart from the SEED_ROW most populated rows
//         else  (or if the above found nothing) start at a random non-zero and take the next unvisited one
//       The frontier is clamped to FRONTIER_SIZE by random eviction, so there is no sorting (much faster preprocessing).
//
// Parameters (env, default = run.sh value): SEED_ROW (50), FRONTIER_SIZE (10000), CACHE_CAPACITY (256), ALPHA (1.0),
// BETA (1.0) (ALPHA / BETA unused by the roulette variant). With PRINT_STATISTICS the simulated cache hits of the chosen
// order are printed, as in the old files.
//
// Rows are split into several runs (COO_REORDER_KEEPS_ROWS = 0, fixedk uses ACCUMULATE mode).

#define COO_REORDER_KEEPS_ROWS  0
#if defined(COO_REORDER_RINGBUFFER_ALL_SEED)
    #define COO_REORDER_NAME    "_RingBufferAllSeed"
#elif defined(COO_REORDER_RINGBUFFER_ROULETTE)
    #define COO_REORDER_NAME    "_RingBufferRoulette"
#else
    #define COO_REORDER_NAME    "_RingBuffer"
#endif

#include "coo_reorder_segment_common.h"

static inline
void
reorder_segments(struct COOArrays * coo)
{
    const int r_first_rows = segment_env_long("SEED_ROW", 50);
    const int frontier_size = segment_env_long("FRONTIER_SIZE", 10000);
    const int cache_capacity = segment_env_long("CACHE_CAPACITY", 256);
    #if !defined(COO_REORDER_RINGBUFFER_ROULETTE)
        const float alpha = segment_env_double("ALPHA", 1.0);
        const float beta = segment_env_double("BETA", 1.0);
    #endif

    INT_T * row_ind = coo->row_ind;
    INT_T * col_ind = coo->col_ind;
    const long m = coo->m;
    const long n = coo->n;

    #ifdef PRINT_STATISTICS
        long total_hit_both = 0, total_hit_row_only = 0, total_hit_col_only = 0, total_hit_none = 0;
    #endif

    segment_order_init(coo);

    #ifdef PRINT_STATISTICS
    #pragma omp parallel reduction(+:total_hit_both, total_hit_row_only, total_hit_col_only, total_hit_none)
    #else
    #pragma omp parallel
    #endif
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];
        long local_nnz = j_e - j_s;

        #ifdef PRINT_STATISTICS
            long local_hit_both = 0;
            long local_hit_row_only = 0;
            long local_hit_col_only = 0;
            long local_hit_none = 0;
        #endif

        if (local_nnz > 0) {
            #if defined(COO_REORDER_RINGBUFFER_ROULETTE)
                // Thread-local random number generator for our stochastic roulette
                unsigned int seed_rng = tnum ^ 0x12345678;
            #endif

            // ---------------------------------------------------------
            // 1. Build local adjacency lists and degree counters
            // ---------------------------------------------------------
            std::unordered_map<long, std::vector<long>> row_nnzs_set;
            std::unordered_map<long, std::vector<long>> col_nnzs_set;
            // Flat arrays for counts since row/col indices are bounded by M and N
            std::vector<int> row_count(m, 0);
            std::vector<int> col_count(n, 0);

            for (long j = j_s; j < j_e; j++) {
                long r = row_ind[j];
                long c = col_ind[j];
                row_nnzs_set[r].push_back(j);
                col_nnzs_set[c].push_back(j);
                row_count[r]++;
                col_count[c]++;
            }

            // ---------------------------------------------------------
            // 2. Find Seeds
            // ---------------------------------------------------------
            std::vector<std::pair<long, int>> pop_rows;
            for (long i = 0; i < m; i++) {
                if (row_count[i] > 0) pop_rows.push_back({i, row_count[i]});
            }
            std::sort(pop_rows.begin(), pop_rows.end(),
                [](const std::pair<long, int>& a, const std::pair<long, int>& b) { return a.second > b.second; });

            #if defined(COO_REORDER_RINGBUFFER_ALL_SEED)
                // Both rows and columns
                std::vector<std::pair<long, int>> pop_cols;
                for (long i = 0; i < n; i++) {
                    if (col_count[i] > 0) pop_cols.push_back({i, col_count[i]});
                }
                std::sort(pop_cols.begin(), pop_cols.end(),
                    [](const std::pair<long, int>& a, const std::pair<long, int>& b) { return a.second > b.second; });

                std::vector<long> seed_rows;
                int seed_limit_r = std::min((int)pop_rows.size(), r_first_rows);
                for (int i = 0; i < seed_limit_r; i++) seed_rows.push_back(pop_rows[i].first);

                std::vector<long> seed_cols;
                // Re-using r_first_rows for columns
                int seed_limit_c = std::min((int)pop_cols.size(), r_first_rows);
                for (int i = 0; i < seed_limit_c; i++) seed_cols.push_back(pop_cols[i].first);
            #else
                std::vector<long> seeds;
                int seed_limit = std::min((int)pop_rows.size(), r_first_rows);
                for (int i = 0; i < seed_limit; i++) seeds.push_back(pop_rows[i].first);
            #endif

            // ---------------------------------------------------------
            // 3. Setup the ring buffers and flat arrays
            // ---------------------------------------------------------
            std::vector<bool> visited(local_nnz, false);
            std::vector<bool> in_frontier(local_nnz, false);

            // The flat arrays for O(1) lookups
            std::vector<char> row_in_cache(m, 0);
            std::vector<char> col_in_cache(n, 0);
            std::vector<int> row_visited_count(m, 0);
            std::vector<int> col_visited_count(n, 0);

            // The ring buffers (size based on physical hardware capacity)
            std::vector<long> row_history(cache_capacity, -1);
            std::vector<long> col_history(cache_capacity, -1);
            int r_head = 0, c_head = 0;

            std::vector<long> reordered_j;
            reordered_j.reserve(local_nnz);
            std::vector<long> frontier;
            size_t MAX_FRONTIER_SIZE = (size_t)frontier_size;

            #if defined(COO_REORDER_RINGBUFFER_ALL_SEED)
                // Pre-fill the frontier with all seeds
                for (long seed_r : seed_rows) {
                    for (long cand_j : row_nnzs_set[seed_r]) {
                        long local_idx = cand_j - j_s;
                        if (!in_frontier[local_idx]) {
                            frontier.push_back(cand_j);
                            in_frontier[local_idx] = true;
                        }
                    }
                }
                for (long seed_c : seed_cols) {
                    for (long cand_j : col_nnzs_set[seed_c]) {
                        long local_idx = cand_j - j_s;
                        if (!in_frontier[local_idx]) {
                            frontier.push_back(cand_j);
                            in_frontier[local_idx] = true;
                        }
                    }
                }
            #endif

            #if defined(COO_REORDER_RINGBUFFER_ROULETTE)
                // The tiered scorer: {Tier [0-3], Coverage Score}
                auto compute_tier = [&](long j) -> std::pair<int, double> {
                    long r = row_ind[j];
                    long c = col_ind[j];

                    bool r_hit = row_in_cache[r];
                    bool c_hit = col_in_cache[c];

                    int tier = 0;
                    if (r_hit && c_hit) tier = 3;       // Golden
                    else if (r_hit) tier = 2;           // Silver (Row prioritize)
                    else if (c_hit) tier = 1;           // Bronze (Col prioritize)
                    else tier = 0;                      // Uncached

                    double n_score = 0.0;
                    if (row_count[r] > 0) n_score += (double)row_visited_count[r] / row_count[r];
                    if (col_count[c] > 0) n_score += (double)col_visited_count[c] / col_count[c];

                    return {tier, n_score};
                };
            #else
                // Heuristic weights
                const double ALPHA = alpha;
                const double BETA  = beta;

                // O(1) scoring lambda
                auto compute_score = [&](long j) -> double {
                    long r = row_ind[j];
                    long c = col_ind[j];

                    // R(v): recency is simply a binary check of the flat arrays
                    double r_score = (row_in_cache[r] ? 1.0 : 0.0) + (col_in_cache[c] ? 1.0 : 0.0);

                    // N(v): coverage uses flat arrays (no hashing)
                    double n_score = 0.0;
                    if (row_count[r] > 0) n_score += (double)row_visited_count[r] / row_count[r];
                    if (col_count[c] > 0) n_score += (double)col_visited_count[c] / col_count[c];

                    return (ALPHA * r_score) + (BETA * n_score);
                };
            #endif

            // ---------------------------------------------------------
            // 4. Main loop
            // ---------------------------------------------------------
            while ((long) reordered_j.size() < local_nnz) {
                long best_candidate = -1;

                #if defined(COO_REORDER_RINGBUFFER_ROULETTE)
                    // Roll the 100-sided dice
                    int roulette = rand_r(&seed_rng) % 100;

                    // --- STRATEGY A: 70% Exploit the Tiers ---
                    if (roulette < 70 && !frontier.empty()) {
                        int highest_tier_found = -1;
                        double best_coverage_in_tier = -1.0;
                        long best_idx = -1;

                        // Scan frontier for the absolute best tier
                        for (size_t i = 0; i < frontier.size(); ) {
                            // If Strategy B or C already processed this node, lazily delete it from the frontier
                            if (visited[frontier[i] - j_s]) {
                                in_frontier[frontier[i] - j_s] = false;
                                frontier[i] = frontier.back();
                                frontier.pop_back();
                                continue; // Do not increment i, we must evaluate the swapped element
                            }

                            auto [tier, coverage] = compute_tier(frontier[i]);

                            if (tier > highest_tier_found) {
                                highest_tier_found = tier;
                                best_coverage_in_tier = coverage;
                                best_idx = i;
                                if (tier == 3) break; // Found a Golden hit, take it immediately
                            } else if (tier == highest_tier_found && coverage > best_coverage_in_tier) {
                                best_coverage_in_tier = coverage;
                                best_idx = i;
                            }
                            i++;
                        }

                        if (best_idx != -1) {
                            best_candidate = frontier[best_idx];
                            frontier[best_idx] = frontier.back();
                            frontier.pop_back();
                            in_frontier[best_candidate - j_s] = false;
                        }
                    }

                    // --- STRATEGY B: 20% Seed Jump ---
                    if (best_candidate == -1 && roulette >= 70 && roulette < 90) {
                        bool found = false;
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
                    }

                    // --- STRATEGY C: 10% Random Unvisited Leap (or absolute fallback) ---
                    if (best_candidate == -1) {
                        // Pick a random index and scan linearly from there until an unvisited node
                        long start_guess = j_s + (rand_r(&seed_rng) % local_nnz);
                        bool found = false;

                        // Scan forward
                        for (long j = start_guess; j < j_e; j++) {
                            if (!visited[j - j_s]) {
                                best_candidate = j;
                                found = true;
                                break;
                            }
                        }
                        // Wrap around and scan backward if needed
                        if (!found) {
                            for (long j = start_guess - 1; j >= j_s; j--) {
                                if (!visited[j - j_s]) {
                                    best_candidate = j;
                                    break;
                                }
                            }
                        }
                    }
                #else
                    // --- A. Score, Sort, Pick, and Clamp ---
                    if (!frontier.empty()) {
                        std::vector<std::pair<double, long>> scored_frontier;
                        scored_frontier.reserve(frontier.size());

                        for (long cand : frontier) {
                            scored_frontier.push_back({compute_score(cand), cand});
                        }

                        std::sort(scored_frontier.begin(), scored_frontier.end(),
                            [](const std::pair<double, long>& a, const std::pair<double, long>& b) {
                                return a.first > b.first;
                            });

                        best_candidate = scored_frontier[0].second;
                        in_frontier[best_candidate - j_s] = false;

                        frontier.clear();
                        size_t keep_count = std::min(scored_frontier.size(), MAX_FRONTIER_SIZE + 1);

                        for (size_t i = 1; i < keep_count; i++) {
                            frontier.push_back(scored_frontier[i].second);
                        }

                        for (size_t i = keep_count; i < scored_frontier.size(); i++) {
                            in_frontier[scored_frontier[i].second - j_s] = false;
                        }
                    }

                    // --- B. Fallback ---
                    if (best_candidate == -1) {
                        #if defined(COO_REORDER_RINGBUFFER_ALL_SEED)
                            // (If the graph is connected this never triggers. If disconnected, it finds the next unvisited chunk)
                            for (long j = j_s; j < j_e; j++) {
                                if (!visited[j - j_s]) {
                                    best_candidate = j;
                                    break;
                                }
                            }
                        #else
                            bool found = false;
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
                            if (!found) {
                                for (long j = j_s; j < j_e; j++) {
                                    if (!visited[j - j_s]) {
                                        best_candidate = j;
                                        break;
                                    }
                                }
                            }
                        #endif
                    }
                #endif

                // --- C. Commit the element and update the ring buffers ---
                visited[best_candidate - j_s] = true;
                reordered_j.push_back(best_candidate);
                long best_r = row_ind[best_candidate];
                long best_c = col_ind[best_candidate];

                #ifdef PRINT_STATISTICS
                    // Evaluate specific cache hits BEFORE modifying the ring buffers
                    bool r_hit = row_in_cache[best_r];
                    bool c_hit = col_in_cache[best_c];
                    if (r_hit && c_hit) local_hit_both++;
                    else if (r_hit) local_hit_row_only++;
                    else if (c_hit) local_hit_col_only++;
                    else local_hit_none++;
                #endif

                row_visited_count[best_r]++;
                col_visited_count[best_c]++;

                // Ring buffer logic (evict old, insert new): rows
                long evict_r = row_history[r_head];
                if (evict_r != -1) row_in_cache[evict_r] = 0; // Evict
                row_history[r_head] = best_r;                 // Insert to history
                row_in_cache[best_r] = 1;                     // Mark active
                r_head = (r_head + 1) % cache_capacity;       // Move head

                // Columns
                long evict_c = col_history[c_head];
                if (evict_c != -1) col_in_cache[evict_c] = 0;
                col_history[c_head] = best_c;
                col_in_cache[best_c] = 1;
                c_head = (c_head + 1) % cache_capacity;

                // --- D. Expand Frontier ---
                for (long neighbor_j : row_nnzs_set[best_r]) {
                    long local_idx = neighbor_j - j_s;
                    if (!visited[local_idx] && !in_frontier[local_idx]) {
                        frontier.push_back(neighbor_j);
                        in_frontier[local_idx] = true;
                    }
                }
                for (long neighbor_j : col_nnzs_set[best_c]) {
                    long local_idx = neighbor_j - j_s;
                    if (!visited[local_idx] && !in_frontier[local_idx]) {
                        frontier.push_back(neighbor_j);
                        in_frontier[local_idx] = true;
                    }
                }

                #if defined(COO_REORDER_RINGBUFFER_ROULETTE)
                    // --- E. Clamp the Frontier (Random Eviction) ---
                    // The scheduler is stochastic, so the frontier doesn't need to be sorted: if it gets too big,
                    // randomly evict down to MAX_FRONTIER_SIZE (O(1) per eviction).
                    while (frontier.size() > MAX_FRONTIER_SIZE) {
                        int drop_idx = rand_r(&seed_rng) % frontier.size();
                        long dropped_cand = frontier[drop_idx];
                        in_frontier[dropped_cand - j_s] = false;

                        frontier[drop_idx] = frontier.back();
                        frontier.pop_back();
                    }
                #endif
            }

            // ---------------------------------------------------------
            // 5. Apply the new ordering
            // ---------------------------------------------------------
            segment_apply_order(coo, j_s, local_nnz, reordered_j.data());
        }

        #ifdef PRINT_STATISTICS
            total_hit_both += local_hit_both;
            total_hit_row_only += local_hit_row_only;
            total_hit_col_only += local_hit_col_only;
            total_hit_none += local_hit_none;
        #endif
    }

    #ifdef PRINT_STATISTICS
        long nnz = coo->nnz;
        printf("====================================================\n");
        printf("TOTAL MATRIX CACHE STATS:\n");
        printf("  Both:     %ld %f%%\n", total_hit_both, (double)total_hit_both / (double)nnz *100);
        printf("  Row Only: %ld %f%%\n", total_hit_row_only, (double)total_hit_row_only / (double)nnz *100);
        printf("  Col Only: %ld %f%%\n", total_hit_col_only, (double)total_hit_col_only / (double)nnz *100);
        printf("  None:     %ld %f%%\n", total_hit_none, (double)total_hit_none / (double)nnz *100);
        printf("====================================================\n");
    #endif
}

#endif /* COO_REORDER_RINGBUFFER_H */
