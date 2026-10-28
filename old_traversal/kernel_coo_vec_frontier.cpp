#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <queue>

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


INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;

ValueType * thread_v_e = NULL;

double * thread_time_compute, * thread_time_barrier;

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

    COOArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k) 
        : Matrix_Format(m, n, nnz, k)
    {
        int num_threads = omp_get_max_threads();
        double time_balance;
        float alpha, beta;
        int row_seed, cross_distance_limit, frontier_size, cache_capacity;

        row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
        col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
        a = (ValueType *) malloc(nnz * sizeof(*a));
        row_seed= atoi(getenv("SEED_ROW"));
        cross_distance_limit = atoi(getenv("CROSS_DISTANCE_LIMIT"));
        frontier_size = atoi(getenv("FRONTIER_SIZE"));
        cache_capacity = atoi(getenv("CACHE_CAPACITY"));
        alpha = atof(getenv("ALPHA"));
        beta = atof(getenv("BETA"));

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

        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
        thread_v_e = (ValueType *) malloc(num_threads * k * sizeof(*thread_v_e));
        
        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                long lower_boundary;
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                #ifdef CUSTOM_COO_VEC_FRONTIER
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


            }
        );
        // printf("COO Vector Cross Preprocessing: Seed Rows=%d, Distance Limit=%d, Cache Capacity=%d, Alpha=%f, Beta=%f\n", row_seed, cross_distance_limit, cache_capacity, alpha, beta);
        this->preprocess_frontier(row_seed, cross_distance_limit, frontier_size, cache_capacity, alpha, beta);

        // Diagnostic (added for the kernel_coo.cpp port check): hash of the final non-zero order, same as in kernel_coo.cpp.
        if (getenv("COO_PRINT_ORDER_HASH") != NULL && atoi(getenv("COO_PRINT_ORDER_HASH"))) {
            uint64_t h = 1469598103934665603ULL;
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
                printf("Thread %ld: nnz range [%d, %d) nnz: %ld of nnz_total: %ld\n", i, thread_j_s[i], thread_j_e[i], thread_j_e[i] - thread_j_s[i], nnz);
            }
        #endif
    }

    ~COOArrays()
    {
        free(a);
        free(row_ind);
        free(col_ind);
        free(thread_j_s);
        free(thread_j_e);

        #ifdef PRINT_STATISTICS
            free(thread_time_barrier);
            free(thread_time_compute);
        #endif
    }

    void preprocess_frontier(int r_first_rows, int cross_distance_limit, int frontier_size, int cache_capacity, float alpha, float beta);
    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

// Forward declarations
void compute_coo_vector_xrow(COOArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_vector_xrow_perfect_nnz_balance(COOArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void COOArrays::preprocess_frontier(int r_first_rows, int cross_distance_limit, int frontier_size, int cache_capacity, float alpha, float beta) 
{
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];
        long local_nnz = j_e - j_s;

        // if (local_nnz <= 0) return;

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
        std::vector<std::pair<long, int>> col_count_vector(col_count.begin(), col_count.end());
        std::sort(col_count_vector.begin(), col_count_vector.end(), 
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

            // R(v): Recency (Linear decay based on how long ago it was seen)
            double r_score = 0.0;
            long r_age = current_step - row_freshness[r];
            long c_age = current_step - col_freshness[c];
            
            //-----------------------------------------------------------------//
            // if (row_freshness[r] > 0 && r_age <= cache_capacity) {
            //     r_score += 1.0 - ((double)r_age / cache_capacity); // Fixed division variable
            // }
            // if (col_freshness[c] > 0 && c_age <= cache_capacity) {
            //     r_score += 1.0 - ((double)c_age / cache_capacity); // Fixed division variable
            // }
            //-----------------------------------------------------------------//
             if (row_freshness[r] > 0 && r_age <= cache_capacity) {
                r_score += 1.0;
            }
            if (col_freshness[c] > 0 && c_age <= cache_capacity) {
                r_score += 1.0;
            }
            //-----------------------------------------------------------------//

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

        // ---------------------------------------------------------
        // 4. Main Greedy Heuristic Loop
        // ---------------------------------------------------------
        while (reordered_j.size() < local_nnz) {
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

        // ---------------------------------------------------------
        // 5. Apply the new ordering to local arrays
        // ---------------------------------------------------------
        std::vector<INT_T> temp_row(local_nnz);
        std::vector<INT_T> temp_col(local_nnz);
        std::vector<ValueType> temp_a(local_nnz);

        for (long i = 0; i < local_nnz; i++) {
            long orig_idx = reordered_j[i];
            temp_row[i] = row_ind[orig_idx];
            temp_col[i] = col_ind[orig_idx];
            temp_a[i]   = a[orig_idx];
        }

        for (long i = 0; i < local_nnz; i++) {
            row_ind[j_s + i] = temp_row[i];
            col_ind[j_s + i] = temp_col[i];
            a[j_s + i]       = temp_a[i];
        }
    }
}

void
COOArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;
    #ifdef CUSTOM_COO_VEC_FRONTIER 
    //     compute_coo_vector_xrow_perfect_nnz_balance(this, x, y, k);
    // #elif CUSTOM_COO_VEC_XROW_ATOMIC || CUSTOM_COO_VEC_XROW_ROW_SPLIT
        // printf("Running COO Vector Cross Kernel with %d threads...\n", omp_get_max_threads());
        compute_coo_vector_xrow(this, x, y, k);
    #endif
}

void
COOArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    compute_coo_sddmm(this, x, y, out, k);
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COOArrays * coo = new COOArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    // printf("Running COO Vector Cross Kernel with %d threads...\n", omp_get_max_threads());
    coo->format_name = (char *) "COO_Frontier";
    return coo;
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
    
    vec_t(VTF, VEC_LEN) v_val, v_x, v_prod, v_y;
    
    c_e_vector = k & mask;
    v_val = vec_set1(VTF, VEC_LEN, val);

    for (c = 0; c < c_e_vector; c += VEC_LEN)
    {
        
        v_x   = vec_loadu(VTF, VEC_LEN, &x[c_idx * k + c]);
        v_prod = vec_mul(VTF, VEC_LEN, v_val, v_x);

        ValueType temp[VEC_LEN];
        vec_storeu(VTF, VEC_LEN, temp, v_prod);
        // vec_storeu(VTF, VEC_LEN, &y[r * k + c], v_prod);
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
    // printf("Thread %d processing j=%ld\n", omp_get_thread_num(), j);
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

//==========================================================================================================================================
//= COO Main Computation Kernels
//==========================================================================================================================================

void
compute_coo_vector_xrow(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    // printf("Running COO Vector XRow Kernel with %d threads...\n", omp_get_max_threads());
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s, j_e, start_row, end_row;
        j_s = thread_j_s[tnum];
        j_e = thread_j_e[tnum];
        // start_row = coo->row_ind[j_s];
        // end_row = coo->row_ind[j_e - 1];

        // printf("Thread %d: Processing NNZ range [%ld, %ld) which corresponds to rows [%ld, %ld]\n", tnum, j_s, j_e, start_row, end_row);
        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

        if (j_e > j_s) {
            start_row = coo->row_ind[j_s];
            end_row = coo->row_ind[j_s];
            for(long j = j_s; j < j_e; j++) {
                if (coo->row_ind[j] < start_row) start_row = coo->row_ind[j];
                if (coo->row_ind[j] > end_row) end_row = coo->row_ind[j];
            }
        }
        else {
            // No nnz assigned to this thread; make the zero-init loop below a no-op.
            start_row = 0;
            end_row = -1;
        }
        
        for (long i=start_row;i<=end_row;i++)
	    {	
        for (long c = 0; c < k; c++)
                y[i*k + c]=0;
        }
        
        for (long j = j_s; j < j_e; j++)
        {
            // Prefetch next values
            // __builtin_prefetch(&coo->row_ind[j+8], 0, 3);
            // __builtin_prefetch(&coo->col_ind[j+8], 0, 3);
            // __builtin_prefetch(&coo->a[j+8], 0, 3);
            // #ifdef CUSTOM_COO_VEC_XROW_ROWIND_LIMIT
            //     coo->row_ind[j] = tnum; // Force each thread to write exclusively to its own L1 cache!
            // #endif
            // #ifdef CUSTOM_COO_VEC_XROW_ROW_SPLIT 
            //     subkernel_val_coo_vec_xrow_noatomic(coo, x, y, j, k);
            // #elif defined(CUSTOM_COO_VEC_XROW_ATOMIC)
            //     subkernel_val_coo_vec_xrow_atomic(coo, x, y, j, k);
            // #endif
            subkernel_val_coo_vec_xrow_noatomic(coo, x, y, j, k);
        }

        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1, _Pragma("omp barrier"));
        thread_time_barrier[tnum] += time;
        #endif
    }
}

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
        // printf("Thread %d: Processing NNZ range [%ld, %ld) which corresponds to rows [%ld, %ld]\n", tnum, j_s, j_e, coo->row_ind[j_s], coo->row_ind[j_e - 1]);
        if (j_e > j_s) {
            // printf("Thread %d: Processing NNZ range [%ld, %ld) which corresponds to rows [%ld, %ld]\n", tnum, j_s, j_e, coo->row_ind[j_s], coo->row_ind[j_e - 1]);
            boundary_row = coo->row_ind[j_e - 1];
            start_row = coo->row_ind[j_s];
            next_start_row = (tnum < num_threads - 1) ? coo->row_ind[thread_j_s[tnum + 1]] : coo->m;
        }


        for(long c = 0; c < k; c++) {
            thread_v_e[tnum * k + c] = 0.0;
        }
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
            long boundary_row = coo->row_ind[j_e_t - 1];

            for (long c = 0; c < k; c++) {
                y[boundary_row * k + c] += thread_v_e[t * k + c];
            }
        }
    }
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