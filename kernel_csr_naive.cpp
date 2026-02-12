#include <stdio.h>
#include <stdlib.h>

#include "macros/cpp_defines.h"

#include "bench_common.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"
#ifdef __cplusplus
}
#endif

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	INT_T * csr_ia;      // the usual rowptr (of size m+1)
	INT_T * csr_ja;      // the colidx of each NNZ (of size nnz)
	ValueType * csr_a;   // the values (of size NNZ)

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
        csr_ia = (typeof(ia)) aligned_alloc(64, (m+1) * sizeof(*ia));
		csr_ja = (typeof(ja)) aligned_alloc(64, nnz * sizeof(*ja));
		csr_a = (typeof(a)) aligned_alloc(64, nnz * sizeof(*a));
		#pragma omp parallel for
		for (long i=0;i<m+1;i++)
			csr_ia[i] = ia[i];
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
		{
			csr_a[i]=a[i];
			csr_ja[i]=ja[i];
        }

	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		free(csr_ja);
		free(csr_ia);
		free(csr_a);
	}

	void spmm(ValueType * x, ValueType * y, int k);
	void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
};

void compute_spmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k);
void compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
CSRArrays::spmm(ValueType * x, ValueType * y, int k)
{
	compute_spmm(this, x, y, k);
}

void
CSRArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
	compute_sddmm(this, x, y, out, k);
}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
	struct CSRArrays * csr = new CSRArrays(row_ptr, col_ind, values, m, n, nnz, k);
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	csr->format_name = (char *) "Basic CSR-CPU";
	return csr;
}

//==========================================================================================================================================
//= Computation
//==========================================================================================================================================

void
compute_spmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	__attribute__((unused)) const ValueType alpha = 1.0;
	__attribute__((unused)) const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
	}

	#ifdef SPMM_KERNEL
		#pragma omp parallel for default(none) shared(csr, x, y, k)
		for (long i = 0; i < csr->m; i++) {
			for (long c = 0; c < k; c++) {
				// ValueType value;
				ValueType sum = 0.0;
				#pragma omp simd reduction(+:sum)
				for (long j = csr->csr_ia[i]; j < csr->csr_ia[i + 1]; j++) {
					// value = csr->csr_a[j] * x[c * csr->n + csr->csr_ja[j]];
					sum += csr->csr_a[j] * x[c * csr->n + csr->csr_ja[j]];
				}
				y[i * k + c] = sum;
			}
		}
	#endif

	if (csr->y == NULL)
	{
		csr->y = y;
	}
}

void
compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k)
{
	__attribute__((unused)) const ValueType alpha = 1.0;
	__attribute__((unused)) const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
		csr->y = y;
	}

	#ifdef SDDMM_KERNEL
		for (i = 0; i < csr->m; i++) {
			for (j = csr->csr_ia[i]; j < csr->csr_ia[i+1]; j++) {
				ValueType value;
				ValueType sum = 0.0;
				long curr_col = csr->csr_ja[j];
				for(long c = 0; c < k; c++) {
					// value = val[j] * x[i*dense_k + k] * y[k*csr_n + curr_col] - compensation;
					// this would also be acceptable, since the values of sparse matrix are all set to 1 for SDDMM.
					value = x[i*k + c] * y[c*csr_n + curr_col];
					sum += value;
				}
				out_gold[j] = sum;
			}
		}
	#endif

	if (csr->out == NULL)
	{
		csr->out = out;
	}
}
