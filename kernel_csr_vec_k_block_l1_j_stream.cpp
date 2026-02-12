#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

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


INT_T * thread_i_s = NULL;
INT_T * thread_i_e = NULL;

INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;

ValueType * thread_v_e = NULL;

double * thread_time_compute, * thread_time_barrier;

// ============================================================================
// CSR Class Definition
// ============================================================================
struct CSRArrays : Matrix_Format
{
    INT_T * ia;      // Row pointers
    INT_T * ja;      // Column indices
    ValueType * a;   // Values

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    long num_loops;

    CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
    {
        int num_threads = omp_get_max_threads();
        
        thread_i_s = (INT_T *) malloc(num_threads * sizeof(*thread_i_s));
        thread_i_e = (INT_T *) malloc(num_threads * sizeof(*thread_i_e));
        thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
        thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
        thread_v_e = (ValueType *) malloc(num_threads * k * sizeof(*thread_v_e));

        #pragma omp parallel
        {
            int tnum = omp_get_thread_num();
            #if defined(NAIVE)
                loop_partitioner_balance_iterations(num_threads, tnum, 0, m, &thread_i_s[tnum], &thread_i_e[tnum]);
            #else
                loop_partitioner_balance_prefix_sums(num_threads, tnum, ia, m, nnz, &thread_i_s[tnum], &thread_i_e[tnum]);
            #endif
        }

        #ifdef PRINT_STATISTICS
            num_loops = 0;
            thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
            thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
        #endif
    }

    ~CSRArrays()
    {
        
        if (thread_i_s) free(thread_i_s);
        if (thread_i_e) free(thread_i_e);
        if (thread_j_s) free(thread_j_s);
        if (thread_j_e) free(thread_j_e);
        if (thread_v_e) free(thread_v_e);

        #ifdef PRINT_STATISTICS
            if (thread_time_barrier) free(thread_time_barrier);
            if (thread_time_compute) free(thread_time_compute);
        #endif
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};

void compute_csr_vector_xrow_k_block_l1(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k);
void compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);


void CSRArrays::spmm(ValueType * x, ValueType * y, int k)
{
    num_loops++;

    compute_csr_vector_xrow_k_block_l1(this, x, y, k);
}

void
CSRArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
	compute_sddmm(this, x, y, out, k);
}

void
compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, __attribute__((unused)) int k)
{
	__attribute__((unused)) const ValueType alpha = 1.0;
	__attribute__((unused)) const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
		csr->y = y;
	}

	if (csr->out == NULL)
	{
		csr->out = out;
	}
}

__attribute__((hot))
static inline
void
subkernel_row_csr_vec_xrow_blocked(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, 
                                  long i, long j, int k_start, int k_end)
{
    long c;
    const long mask = ~(((long) VEC_LEN) - 1);
    vec_t(VTF, VEC_LEN) v_a, v_x, v_sum;
    
    int k_chunk = k_end - k_start;
    long c_e_vector = k_start + (k_chunk & mask);


    v_a = vec_set1(VTF, VEC_LEN, csr->a[j]); 
    

    for (c = k_start; c < c_e_vector; c += VEC_LEN)
    {
        v_sum = vec_loadu(VTF, VEC_LEN, &y[i * csr->k + c]);
        v_x = vec_loadu(VTF, VEC_LEN, &x[csr->ja[j] * csr->k + c]);
        v_sum = vec_fmadd(VTF, VEC_LEN, v_a, v_x, v_sum);
        vec_storeu(VTF, VEC_LEN, &y[i * csr->k + c], v_sum);
    }
    

    for (c = c_e_vector; c < k_end; c++) {
        y[i * csr->k + c] += csr->a[j] * x[csr->ja[j] * csr->k + c];
    }
}


void
subkernel_csr_vec_xrow_blocked(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, 
                              long i_s, long i_e, int k, int block_size_k, int block_size_i)
{
    long i, j, j_s, j_e, kb, ib;

    for (ib = i_s; ib < i_e; ib += block_size_i) {
        
        long i_limit = (ib + block_size_i > i_e) ? i_e : ib + block_size_i;

        for (kb = 0; kb < k; kb += block_size_k) {
            
            int k_end = (kb + block_size_k > k) ? k : kb + block_size_k;

            for (i = ib; i < i_limit; i++) {
                j_s = csr->ia[i];
                j_e = csr->ia[i+1];


                for (j = j_s; j < j_e; j++) {
                    subkernel_row_csr_vec_xrow_blocked(csr, x, y, i, j, kb, k_end);
                }
            }
        }
    }
}


// ============================================================================
// MAIN COMPUTE FUNCTION
// ============================================================================
void
compute_csr_vector_xrow_k_block_l1(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{

    int l2_floats = atoi(getenv("L2_FLOATS")); 
    if (l2_floats <= 0) l2_floats = 32000; 
    

    int block_size_k = 256;


    // int block_size_i = l2_floats / k;
    int block_size_i = 512; 

    if (block_size_i < 1) block_size_i = 1;
    // printf("Computed block sizes: T_i = %d, T_k = %d (L2 floats: %d)\n", block_size_i, block_size_k, l2_floats);

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();
        long i_s, i_e;
        
        i_s = thread_i_s[tnum];
        i_e = thread_i_e[tnum];

        #ifdef PRINT_STATISTICS
        double time = time_it(1,
        #endif

            subkernel_csr_vec_xrow_blocked(csr, x, y, i_s, i_e, k, block_size_k, block_size_i);
        
        #ifdef PRINT_STATISTICS
        );
        thread_time_compute[tnum] += time;
        time = time_it(1,
            _Pragma("omp barrier")
        );
        thread_time_barrier[tnum] += time;
        #endif
    }
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    struct CSRArrays * csr = new CSRArrays(row_ptr, col_ind, values, m, n, nnz, k);
    csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
    
    #if defined(CUSTOM_CSR_VEC_XROW_BLOCKED_L1_JSTREAM)
        csr->format_name = (char *) "Custom_CSR_VEC_XROW_BLOCKED_L1_JSTREAM";
    #endif
    
    return csr;
}

void CSRArrays::statistics_start()
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

int CSRArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
    return 0;
}