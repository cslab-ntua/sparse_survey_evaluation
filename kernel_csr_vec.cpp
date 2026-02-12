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
		// #define VEC_LEN  1
		#define VEC_LEN  vec_len_default_f32
		// #define VEC_LEN  vec_len_default_f64
		// #define VEC_LEN  4
		// #define VEC_LEN  8
		// #define VEC_LEN  16
		// #define VEC_LEN  32
	#elif DOUBLE == 1
		#define VTI   i64
		#define VTF   f64
		#define VTM   m64
		#define VEC_SCALE_SHIFT  3
		#define VEC_LEN  vec_len_default_f64
		// #define VEC_LEN  1
	#endif

	#include "vectorization/vectorization_gen.h"
#ifdef __cplusplus
}
#endif


INT_T * thread_i_s = NULL;
INT_T * thread_i_e = NULL;

INT_T * thread_j_s = NULL;
INT_T * thread_j_e = NULL;

// ValueType * thread_v_s = NULL;
ValueType * thread_v_e = NULL;

int prefetch_distance = 32;

double * thread_time_compute, * thread_time_barrier;


struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	long num_loops;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		int num_threads = omp_get_max_threads();
		double time_balance;

		// ia = (typeof(ia)) aligned_alloc(64, (m+1) * sizeof(*ia));
		// ja = (typeof(ja)) aligned_alloc(64, nnz * sizeof(*ja));
		// a = (typeof(a)) aligned_alloc(64, nnz * sizeof(*a));
		// #pragma omp parallel for
		// for (long i=0;i<m+1;i++)
		// 	ia[i] = row_ptr_in[i];
		// #pragma omp parallel for
		// for(long i=0;i<nnz;i++)
		// {
		// 	a[i]=values[i];
		// 	ja[i]=col_ind[i];
		// }

		thread_i_s = (INT_T *) malloc(num_threads * sizeof(*thread_i_s));
		thread_i_e = (INT_T *) malloc(num_threads * sizeof(*thread_i_e));
		thread_j_s = (INT_T *) malloc(num_threads * sizeof(*thread_j_s));
		thread_j_e = (INT_T *) malloc(num_threads * sizeof(*thread_j_e));
		
		// thread_v_s = (ValueType *) malloc(num_threads * sizeof(*thread_v_s));
		thread_v_e = (ValueType *) malloc(num_threads * k * sizeof(*thread_v_e));
		// printf("before loop partitioning: using %d threads\n", num_threads);
		time_balance = time_it(1,
			_Pragma("omp parallel")
			{
				int tnum = omp_get_thread_num();
				// printf("Thread %d starting loop partitioning\n", tnum);
				#if defined(NAIVE)
					loop_partitioner_balance_iterations(num_threads, tnum, 0, m, &thread_i_s[tnum], &thread_i_e[tnum]);
				#else
					// int use_processes = atoi(getenv("USE_PROCESSES"));
					// if (use_processes)
					// {
					// 	loop_partitioner_balance_iterations(num_threads, tnum, 0, m, &thread_i_s[tnum], &thread_i_e[tnum]);
					// }
					// else
					// {
                    #ifdef CUSTOM_VECTOR_PERFECT_NNZ_BALANCE
                        long lower_boundary;
                        // long higher_boundary;
                        loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &thread_j_s[tnum], &thread_j_e[tnum]);
                        macros_binary_search(ia, 0, m, thread_j_s[tnum], &lower_boundary, NULL);           // Index boundaries are inclusive.
                        thread_i_s[tnum] = lower_boundary;
                        _Pragma("omp barrier")
                        if (tnum == num_threads - 1)   // If we calculate each thread's boundaries individually some empty rows might be unassigned.
                            thread_i_e[tnum] = m;
                        else
                            thread_i_e[tnum] = thread_i_s[tnum+1] + 1;
                    #else
                        loop_partitioner_balance_prefix_sums(num_threads, tnum, ia, m, nnz, &thread_i_s[tnum], &thread_i_e[tnum]);
                    #endif
					// }
				#endif
			}
		);
		// printf("balance time = %g\n", time_balance);

		#ifdef PRINT_STATISTICS
			long i;
			num_loops = 0;
			thread_time_barrier = (double *) malloc(num_threads * sizeof(*thread_time_barrier));
			thread_time_compute = (double *) malloc(num_threads * sizeof(*thread_time_compute));
			for (i=0;i<num_threads;i++)
			{
				long rows, nnz;
				INT_T i_s, i_e, j_s, j_e;
				i_s = thread_i_s[i];
				i_e = thread_i_e[i];
				j_s = thread_j_s[i];
				j_e = thread_j_e[i];
				rows = i_e - i_s;
				nnz = ia[i_e] - ia[i_s];
				printf("%10ld: rows=[%10d(%10d), %10d(%10d)] : %10ld(%10ld)   ,   nnz=[%10d, %10d]:%10d\n", i, i_s, ia[i_s], i_e, ia[i_e], rows, nnz, j_s, j_e, j_e-j_s);
			}
		#endif
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);
		free(thread_i_s);
		free(thread_i_e);
		free(thread_j_s);
		free(thread_j_e);
		// free(thread_v_s);
		free(thread_v_e);

		#ifdef PRINT_STATISTICS
			free(thread_time_barrier);
			free(thread_time_compute);
		#endif
	}

	void spmm(ValueType * x, ValueType * y, int k);
	void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr_vector(CSRArrays * restrict csr, ValueType * restrict x , ValueType * restrict y, int k);
void compute_csr_vector_xrow(CSRArrays * restrict csr, ValueType * restrict x , ValueType * restrict y, int k);
void compute_csr_vector_xrow_prefetch(CSRArrays * restrict csr, ValueType * restrict x , ValueType * restrict y, int k);
void compute_csr_vector_perfect_nnz_balance(CSRArrays * restrict csr, ValueType * restrict x , ValueType * restrict y, int k);
void compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
CSRArrays::spmm(ValueType * x, ValueType * y, int k)
{
	num_loops++;
	#if defined(CUSTOM_CSR_VEC)
		compute_csr_vector(this, x, y, k);
    #elif defined(CUSTOM_CSR_VEC_XROW)
		compute_csr_vector_xrow(this, x, y, k);
	#elif defined(CUSTOM_CSR_VEC_XROW_PREFETCH)
		compute_csr_vector_xrow_prefetch(this, x, y, k);
	#elif defined(CUSTOM_VECTOR_PERFECT_NNZ_BALANCE)
		compute_csr_vector_perfect_nnz_balance(this, x, y, k);
	#endif
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

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
	// if (symmetric && !symmetry_expanded)
	// 	error("symmetric matrices have to be expanded to be supported by this format");
	struct CSRArrays * csr = new CSRArrays(row_ptr, col_ind, values, m, n, nnz, k);
	// printf("Created CSR format struct\n");
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	#if defined(CUSTOM_CSR_VEC)
		csr->format_name = (char *) "Custom_CSR_VEC";
    #elif defined(CUSTOM_CSR_VEC_XROW)
		csr->format_name = (char *) "Custom_CSR_VEC_XROW";
	#elif defined(CUSTOM_CSR_VEC_XROW_PREFETCH)
		csr->format_name = (char *) "Custom_CSR_VEC_XROW_PREFETCH";
	#elif defined(CUSTOM_VECTOR_PERFECT_NNZ_BALANCE)
		csr->format_name = (char *) "Custom_CSR_PBV_VEC";
	// #elif defined(CUSTOM_VECTOR_PERFECT_NNZ_BALANCE_PREFETCH)
	// 	csr->format_name = (char *) "Custom_CSR_PBVPrefetch_VEC";
	#endif
	return csr;
}


//==========================================================================================================================================
//= Subkernels Single Row CSR
//==========================================================================================================================================


__attribute__((hot,pure))
static inline
double
subkernel_row_csr_vec(CSRArrays * restrict csr, ValueType * restrict x, long j_s, long j_e, int c)
{
    long j, j_e_vector;
	const long mask = ~(((long) VEC_LEN) - 1); // Minimum number of elements for the vectorized code (power of 2).
	vec_t(VTF, VEC_LEN) v_a, v_x, v_sum;
	ValueType sum = 0;
	v_sum = vec_set1(VTF, VEC_LEN, 0);
	sum = 0;
	j_e_vector = j_s + ((j_e - j_s) & mask);
	for (j=j_s;j<j_e_vector;j+=VEC_LEN)
	{
		v_a = vec_loadu(VTF, VEC_LEN, &csr->a[j]);
		v_x = vec_set_iter(VTF, VEC_LEN, iter, x[c * csr->n + csr->ja[j+iter]]);
		v_sum = vec_fmadd(VTF, VEC_LEN, v_a, v_x, v_sum);
	}
	sum = vec_reduce_add(VTF, VEC_LEN, v_sum);
	for (j=j_e_vector;j<j_e;j++)
		sum += csr->a[j] * x[c * csr->n + csr->ja[j]];
	return sum;
}

__attribute__((hot))
static inline
void
subkernel_row_csr_vec_xrow(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i, long j, int k)
{
    // printf("    Processing nnz index %ld\n", j);
    long c, c_e_vector;
	const long mask = ~(((long) VEC_LEN) - 1); // Minimum number of elements for the vectorized code (power of 2).
	vec_t(VTF, VEC_LEN) v_a, v_x, v_sum;
	// ValueType sum = 0;
	// v_sum = vec_set1(VTF, VEC_LEN, 0);
	// sum = 0;
	c_e_vector = k & mask;
	v_a = vec_set1(VTF, VEC_LEN, csr->a[j]);
	// #pragma GCC unroll 4
	for (c=0;c<c_e_vector;c+=VEC_LEN)
	{
        v_sum = vec_loadu(VTF, VEC_LEN, &y[i* k + c]);
        v_x = vec_loadu(VTF, VEC_LEN, &x[csr->ja[j]*k + c]);
		// v_x = vec_set_iter(VTF, VEC_LEN, iter, x[csr->ja[j]*csr->n + c+iter]);
		v_sum = vec_fmadd(VTF, VEC_LEN, v_a, v_x, v_sum);
        vec_storeu(VTF, VEC_LEN, &y[i* k + c], v_sum);
	}
	// printf("    Processing up to column %d in vectorized way\n", k);
	for (c=c_e_vector;c<k;c++){
        // printf("    Processing column %ld\n", c);
		y[i* k + c] += csr->a[j] * x[csr->ja[j]* k + c];
    }
}

__attribute__((hot))
static inline
void
subkernel_row_csr_vec_xrow_prefetch(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i, long j, int k)
{
    // printf("    Processing nnz index %ld\n", j);
    long c, c_e_vector;
	const long mask = ~(((long) VEC_LEN) - 1); // Minimum number of elements for the vectorized code (power of 2).
	vec_t(VTF, VEC_LEN) v_a, v_x, v_sum;
	// ValueType sum = 0;
	// v_sum = vec_set1(VTF, VEC_LEN, 0);
	// sum = 0;
	c_e_vector = k & mask;
	// #pragma GCC unroll 4
	for (c=0;c<c_e_vector;c+=VEC_LEN)
	{
		__builtin_prefetch(&x[csr->ja[j]*k + c], 0, 3);
		__builtin_prefetch(&x[csr->ja[j]*k + c], 0, 3);
        v_sum = vec_loadu(VTF, VEC_LEN, &y[i* k + c]);
		v_a = vec_set1(VTF, VEC_LEN, csr->a[j]);
        v_x = vec_loadu(VTF, VEC_LEN, &x[csr->ja[j]*k + c]);
		// v_x = vec_set_iter(VTF, VEC_LEN, iter, x[csr->ja[j]*csr->n + c+iter]);
		v_sum = vec_fmadd(VTF, VEC_LEN, v_a, v_x, v_sum);
        vec_storeu(VTF, VEC_LEN, &y[i* k + c], v_sum);
	}
	// printf("    Processing up to column %d in vectorized way\n", k);
	for (c=c_e_vector;c<k;c++){
        // printf("    Processing column %ld\n", c);
		y[i* k + c] += csr->a[j] * x[csr->ja[j]* k + c];
    }
}



void
subkernel_csr_vec(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e, int k)
{
	ValueType sum;
	long i, j, j_s, j_e;
	for (i=i_s;i<i_e;i++)
	{	
        j_s = csr->ia[i];
		j_e = csr->ia[i+1];
		for (long c = 0; c < k; c++) 
		{
			y[i * k + c] = subkernel_row_csr_vec(csr, x, j_s, j_e, c);
		}
	}
}

void
subkernel_csr_vec_xrow(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e, int k)
{
	ValueType sum;
	long i, j, j_s, j_e;
	for (i=i_s;i<i_e;i++)
	{	
        for (long c = 0; c < k; c++)
                y[i*k + c]=0;
        j_s = csr->ia[i];
		j_e = csr->ia[i+1];
		for (j=j_s;j<j_e;j++)
		{
            // printf("Processing row %ld, nnz index %ld\n", i, j);
			subkernel_row_csr_vec_xrow(csr, x, y, i, j, k);
		}
	}
}

void
subkernel_csr_vec_xrow_prefetch(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, long i_s, long i_e, int k)
{
	ValueType sum;
	long i, j, j_s, j_e;
	for (i=i_s;i<i_e;i++)
	{	
        for (long c = 0; c < k; c++)
                y[i*k + c]=0;
        j_s = csr->ia[i];
		j_e = csr->ia[i+1];
		for (j=j_s;j<j_e;j++)
		{
            // printf("Processing row %ld, nnz index %ld\n", i, j);
			__builtin_prefetch(&csr->a[j], 0, 3);
			subkernel_row_csr_vec_xrow_prefetch(csr, x, y, i, j, k);
		}
	}
}

//==========================================================================================================================================
//= CSR Custom Vector
//==========================================================================================================================================


void
compute_csr_vector(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		long i_s, i_e;
		i_s = thread_i_s[tnum];
		i_e = thread_i_e[tnum];
		#ifdef PRINT_STATISTICS
		double time;
		time = time_it(1,
		#endif
			subkernel_csr_vec(csr, x, y, i_s, i_e, k);
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

void
compute_csr_vector_xrow(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		long i_s, i_e;
		i_s = thread_i_s[tnum];
		i_e = thread_i_e[tnum];
		#ifdef PRINT_STATISTICS
		double time;
		time = time_it(1,
		#endif
			subkernel_csr_vec_xrow(csr, x, y, i_s, i_e, k);
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

void
compute_csr_vector_xrow_prefetch(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		long i_s, i_e;
		i_s = thread_i_s[tnum];
		i_e = thread_i_e[tnum];
		#ifdef PRINT_STATISTICS
		double time;
		time = time_it(1,
		#endif
			subkernel_csr_vec_xrow_prefetch(csr, x, y, i_s, i_e, k);
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



//==========================================================================================================================================
//= CSR Custom Perfect NNZ Balance Vector
//==========================================================================================================================================


void
compute_csr_vector_perfect_nnz_balance(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	int num_threads = omp_get_max_threads();
	long t;
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		long i, i_s, i_e, j, j_s, j_e;

		i_s = thread_i_s[tnum];
		i_e = thread_i_e[tnum];

		for (long c = 0; c < k; c++) 
		{
			if (i_e - 1 >= 0)
				y[(i_e - 1)* k + c] = 0;
		}

		ValueType sum;
		j_s = thread_j_s[tnum];

		j = j_s;
		for (i=i_s;i<i_e-1;i++)
		{

			j_e = csr->ia[i+1];
			for (long c = 0; c < k; c++)
			{
				y[i * k + c] = subkernel_row_csr_vec(csr, x, j_s, j_e, c);

			}
			j = j_e;
		}

		i = i_e - 1;
		for (long c = 0; c < k; c++) 
		{
			j =  csr->ia[i];
			if (j_s > j)
				j = j_s;
			j_e = thread_j_e[tnum];
			sum = 0;
			for (;j<j_e;j++)
			{
				
				sum += csr->a[j] * x[c * csr->n + csr->ja[j]];
			}
			thread_v_e[tnum * k + c] = sum;
		}
	}
	for (t=0;t<num_threads;t++)
	{
		for (long c = 0; c < k; c++) 
		{
			if (thread_i_e[t] - 1 < csr->m)
				y[(thread_i_e[t] - 1) * k + c] += thread_v_e[t * k + c];
		}
	}
}



//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSRArrays::statistics_start()
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
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSRArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{

	return 0;
}