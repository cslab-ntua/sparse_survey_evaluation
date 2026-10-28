#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <algorithm>

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

// ==========================================================================================================================================
// = RCM Utilities
// ==========================================================================================================================================

struct DegreeNode {
    INT_T id;
    INT_T degree;
    bool operator<(const DegreeNode& other) const {
        if (degree != other.degree) return degree < other.degree;
        return id < other.id; // Tie-breaker for deterministic output
    }
};

struct COOElement {
    INT_T r;
    INT_T c;
    ValueType v;
    long old_nnz_id; // FIX 2: Track original position for SDDMM mapping
    bool operator<(const COOElement& other) const {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};

// Generates RCM Permutation with Automatic Symmetrization
void generate_rcm(long m, INT_T* csr_ia, INT_T* csr_ja, INT_T* perm, INT_T* inv_perm) {
    // 1. Fast Structural Symmetry Check
    bool is_symmetric = true;
    for (long i = 0; i < m; i++) {
        for (long j = csr_ia[i]; j < csr_ia[i+1]; j++) {
            long col = csr_ja[j];
            if (col == i) continue; // Ignore diagonal
            
            // Binary search for reverse edge
            long left = csr_ia[col], right = csr_ia[col+1] - 1;
            bool found = false;
            while (left <= right) {
                long mid = left + (right - left) / 2;
                if (csr_ja[mid] == i) { found = true; break; }
                else if (csr_ja[mid] < i) left = mid + 1;
                else right = mid - 1;
            }
            if (!found) { is_symmetric = false; break; }
        }
        if (!is_symmetric) break;
    }

    // 2. Build A + A^T if asymmetric
    std::vector<std::vector<INT_T>> adj(m);
    if (!is_symmetric) {
        // printf("[RCM] Matrix is asymmetric. Applying A + A^T symmetrization...\n");
        for (long i = 0; i < m; i++) {
            for (long j = csr_ia[i]; j < csr_ia[i+1]; j++) {
                long col = csr_ja[j];
                adj[i].push_back(col);
                if (col != i) adj[col].push_back(i); // Add reverse edge
            }
        }
        // Remove duplicates
        for (long i = 0; i < m; i++) {
            std::sort(adj[i].begin(), adj[i].end());
            adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        }
    } 
    // else {
    //     printf("[RCM] Matrix is structurally symmetric. Skipping A + A^T...\n");
    // }

    auto get_degree = [&](long node) -> INT_T {
        return is_symmetric ? (csr_ia[node+1] - csr_ia[node]) : adj[node].size();
    };

    // 3. Pre-sort all nodes by degree.
    // This solves the disconnected components bug by guaranteeing 
    // the BFS safely picks up the lowest-degree unvisited node every time.
    std::vector<DegreeNode> all_nodes(m);
    for (long i = 0; i < m; i++) {
        all_nodes[i] = { (INT_T)i, get_degree(i) };
    }
    std::sort(all_nodes.begin(), all_nodes.end());

    // 4. Breadth-First Search (Cuthill-McKee)
    std::vector<bool> visited(m, false);
    long perm_idx = 0;

    // Iterate through the pre-sorted list to ensure NO isolated components are skipped
    for (long k = 0; k < m; k++) {
        long start_node = all_nodes[k].id;
        if (visited[start_node]) continue;

        std::queue<INT_T> q;
        q.push(start_node);
        visited[start_node] = true;

        while (!q.empty()) {
            INT_T curr = q.front();
            q.pop();
            perm[perm_idx++] = curr; // old_id = perm[new_id]

            std::vector<DegreeNode> neighbors;
            if (!is_symmetric) {
                for (INT_T neighbor : adj[curr]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        neighbors.push_back({neighbor, get_degree(neighbor)});
                    }
                }
            } else {
                for (long j = csr_ia[curr]; j < csr_ia[curr+1]; j++) {
                    INT_T neighbor = csr_ja[j];
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        neighbors.push_back({neighbor, get_degree(neighbor)});
                    }
                }
            }

            // Sort neighbors by degree to minimize bandwidth
            std::sort(neighbors.begin(), neighbors.end());
            for (const auto& n : neighbors) {
                q.push(n.id);
            }
        }
    }

    // 5. Reverse it to get RCM!
    std::reverse(perm, perm + m);

    // 6. Build Inverse Permutation map
    for (long i = 0; i < m; i++) {
        inv_perm[perm[i]] = i; // new_id = inv_perm[old_id]
    }
}

// ==========================================================================================================================================
// = COORCMArrays Structure
// ==========================================================================================================================================

struct COORCMArrays : Matrix_Format
{
    INT_T * row_ind; 
    INT_T * col_ind; 
    ValueType * a;   
    long * nnz_perm; // FIX 2: Store mapping for SDDMM un-shuffle

    // Permutation Maps
    INT_T * perm;
    INT_T * inv_perm;

    // Internal execution buffers
    ValueType * x_rcm = NULL;
    ValueType * y_rcm = NULL;
    ValueType * out_rcm = NULL; // FIX 2: Temporary SDDMM output buffer

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    long num_loops;

    COORCMArrays(INT_T * csr_ia, INT_T * csr_ja, ValueType * csr_a, long m, long n, long nnz, int k) 
        : Matrix_Format(m, n, nnz, k)
    {
        int num_threads = omp_get_max_threads();
        double time_balance;

        row_ind = (INT_T *) malloc(nnz * sizeof(*row_ind));
        col_ind = (INT_T *) malloc(nnz * sizeof(*col_ind));
        a = (ValueType *) malloc(nnz * sizeof(*a));
        nnz_perm = (long *) malloc(nnz * sizeof(*nnz_perm));
        out_rcm = (ValueType *) malloc(nnz * sizeof(*out_rcm));
        
        perm = (INT_T *) malloc(m * sizeof(*perm));
        inv_perm = (INT_T *) malloc(m * sizeof(*inv_perm));

        x_rcm = (ValueType *) malloc(m * k * sizeof(*x_rcm));
        y_rcm = (ValueType *) malloc(m * k * sizeof(*y_rcm));

        // 1. Generate RCM Reordering Map
        generate_rcm(m, csr_ia, csr_ja, perm, inv_perm);

        // 2. Relabel coordinates and sort to maintain strictly linear row-major execution
        std::vector<COOElement> elements(nnz);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (long i = 0; i < m; i++) {
            for (long j = csr_ia[i]; j < csr_ia[i+1]; j++) {
                elements[j].r = inv_perm[i];          // Apply new RCM Row ID
                elements[j].c = inv_perm[csr_ja[j]];  // Apply new RCM Col ID
                
                #ifdef CUSTOM_COO_VEC_XROW_COLIND0
                    elements[j].c = inv_perm[csr_ja[j]]; // FIX 3: Changed from [0] to [j]
                #endif
                elements[j].v = csr_a[j];
                elements[j].old_nnz_id = j; // FIX 2: Track it
            }
        }

        std::sort(elements.begin(), elements.end());

        // 3. Load sorted data back into parallel arrays
        #pragma omp parallel for schedule(static)
        for(long j = 0; j < nnz; j++) {
            row_ind[j] = elements[j].r;
            col_ind[j] = elements[j].c;
            a[j] = elements[j].v;
            nnz_perm[j] = elements[j].old_nnz_id; // FIX 2: Store mapping
        }

        // 4. Partitioning (Runs flawlessly on the sorted RCM arrays)
        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
        thread_v_e = (ValueType *) malloc(num_threads * k * sizeof(*thread_v_e));
        
        time_balance = time_it(1,
            _Pragma("omp parallel")
            {
                int tnum = omp_get_thread_num();
                loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);

                #if defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT) || defined(CUSTOM_COO_VEC_XROW_COLIND0) || defined(CUSTOM_COO_VEC_RCM)
                    if (tnum > 0 && thread_j_s[tnum] < nnz) {
                        while(thread_j_s[tnum] < nnz && row_ind[thread_j_s[tnum]] == row_ind[thread_j_s[tnum] - 1]) {
                            thread_j_s[tnum]++;
                        }
                    }
                    
                    _Pragma("omp barrier")
                    
                    if (tnum == num_threads - 1) {
                        thread_j_e[tnum] = nnz;
                    } else {
                        thread_j_e[tnum] = thread_j_s[tnum + 1];
                    }
                    
                    if (thread_j_e[tnum] < thread_j_s[tnum]) {
                        thread_j_e[tnum] = thread_j_s[tnum];
                    }
                #endif
            }
        );

        #ifdef PRINT_STATISTICS
            long i;
            num_loops = 0;
            thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
            thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
        #endif
    }

    ~COORCMArrays()
    {
        free(a); free(row_ind); free(col_ind);
        free(nnz_perm); free(out_rcm);
        free(thread_j_s); free(thread_j_e); free(thread_v_e);
        free(perm); free(inv_perm);
        free(x_rcm); free(y_rcm);
        #ifdef PRINT_STATISTICS
            free(thread_time_barrier); free(thread_time_compute);
        #endif
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
    void statistics_start();
    int statistics_print_data(char * buf, long buf_n);
};

// Forward declarations
void compute_coo_vector_xrow(COORCMArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_vector_xrow_perfect_nnz_balance(COORCMArrays * restrict coo, ValueType * restrict x , ValueType * restrict y, int k);
void compute_coo_sddmm(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

// ==========================================================================================================================================
// = Interface Implementations (WITH RCM PERMUTATION LOGIC)
// ==========================================================================================================================================

void
COORCMArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;

    // 1. Shuffle X into the optimized RCM layout
    #pragma omp parallel for schedule(static)
    for (long new_i = 0; new_i < m; new_i++) {
        long old_i = perm[new_i];
        for (long c = 0; c < k; c++) {
            x_rcm[new_i * k + c] = x[old_i * k + c];
        }
    }

    // FIX 1: Globally initialize y_rcm to prevent garbage data from empty rows
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < m * k; i++) {
        y_rcm[i] = 0.0;
    }

    // 2. Execute highly optimized SpMM kernel
    #if defined(CUSTOM_COO_VEC_XROW_PERFECT_NNZ_BALANCE)
        compute_coo_vector_xrow_perfect_nnz_balance(this, x_rcm, y_rcm, k);
    #elif defined(CUSTOM_COO_VEC_XROW_ATOMIC) || defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT) || defined(CUSTOM_COO_VEC_XROW_COLIND0) || defined(CUSTOM_COO_VEC_RCM)
        compute_coo_vector_xrow(this, x_rcm, y_rcm, k);
    #endif

    // 3. Un-shuffle Y_rcm back into the expected, correct Y format
    #pragma omp parallel for schedule(static)
    for (long old_i = 0; old_i < m; old_i++) {
        long new_i = inv_perm[old_i];
        for (long c = 0; c < k; c++) {
            y[old_i * k + c] = y_rcm[new_i * k + c];
        }
    }
}

void
COORCMArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    // FIX 2: Shuffle Dense Inputs for SDDMM
    #pragma omp parallel for schedule(static)
    for (long new_i = 0; new_i < m; new_i++) {
        long old_i = perm[new_i];
        for (long c = 0; c < k; c++) {
            x_rcm[new_i * k + c] = x[old_i * k + c];
            y_rcm[new_i * k + c] = y[old_i * k + c]; 
        }
    }

    // Execute SDDMM kernel using RCM arrays and write to out_rcm
    compute_coo_sddmm(this, x_rcm, y_rcm, out_rcm, k);

    // FIX 2: Un-shuffle SDDMM output back to the original Non-Zero order
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < nnz; j++) {
        long old_j = nnz_perm[j];
        out[old_j] = out_rcm[j];
    }
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct COORCMArrays * coo = new COORCMArrays(row_ptr, col_ind, values, m, n, nnz, k);
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    coo->format_name = (char *) "COO_RCM_Vec";
    return coo;
}

// ==========================================================================================================================================
// = Subkernels COO
// ==========================================================================================================================================

__attribute__((hot))
static inline
void
subkernel_val_coo_vec_xrow_atomic(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
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
subkernel_val_coo_vec_xrow_noatomic(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, long j, int k)
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
subkernel_val_coo_vec_xrow_partial(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict out_buf, long j, int k)
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
subkernel_val_coo_sddmm(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, long j, int k)
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
compute_coo_vector_xrow(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s, j_e;
        j_s = thread_j_s[tnum];
        j_e = thread_j_e[tnum];
        
        if (j_e > j_s) {
            #ifdef PRINT_STATISTICS
            double time = time_it(1,
            #endif

            // FIX 1: Removed local y_rcm zeroing loops to fix garbage data bug
            
            for (long j = j_s; j < j_e; j++) {
                #if defined(CUSTOM_COO_VEC_XROW_ROW_SPLIT) || defined(CUSTOM_COO_VEC_XROW_COLIND0) || defined(CUSTOM_COO_VEC_RCM)
                    subkernel_val_coo_vec_xrow_noatomic(coo, x, y, j, k);
                #elif defined(CUSTOM_COO_VEC_XROW_ATOMIC)
                    subkernel_val_coo_vec_xrow_atomic(coo, x, y, j, k);
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
}

void
compute_coo_vector_xrow_perfect_nnz_balance(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    int num_threads = omp_get_max_threads();
    
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long j_s = thread_j_s[tnum];
        long j_e = thread_j_e[tnum];
        
        long boundary_row = -1;
        long start_row = -1;
        long next_start_row = -1; 
        
        if (j_e > j_s) {
            boundary_row = coo->row_ind[j_e - 1];
            start_row = coo->row_ind[j_s];
            next_start_row = (tnum < num_threads - 1) ? coo->row_ind[thread_j_s[tnum + 1]] : coo->m;
        }

        for(long c = 0; c < k; c++) {
            thread_v_e[tnum * k + c] = 0.0;
        }

        // FIX 1: Removed local y_rcm zeroing loops here as well

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
compute_coo_sddmm(COORCMArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k)
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
COORCMArrays::statistics_start()
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
COORCMArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}