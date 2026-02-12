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

#include <mkl.h>

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	sparse_matrix_t A;
	matrix_descr descr;
	const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
	const sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		const sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
		const sparse_memory_usage_t policy = SPARSE_MEMORY_NONE;
		const int expected_calls = 128;

		descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		mkl_verbose(1);
		#if DOUBLE == 0
			mkl_sparse_s_create_csr(&A, indexing, m, n, ia, ia+1, ja, a);
		#elif DOUBLE == 1
			mkl_sparse_d_create_csr(&A, indexing, m, n, ia, ia+1, ja, a);
		#endif

		// Sort the columns
		mkl_sparse_order(A);

		// execute preprocess
		mkl_sparse_set_mv_hint(A, operation, descr, expected_calls);
		mkl_sparse_set_memory_hint(A, policy);
		mkl_sparse_optimize(A);
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		mkl_sparse_destroy(A);
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
	csr->format_name = (char *) "MKL_IE";
	return csr;
}

//==========================================================================================================================================
//= Computation
//==========================================================================================================================================

void
compute_spmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	const ValueType alpha = 1.0;
	const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
	}

	#if DOUBLE == 0
		mkl_sparse_s_mm(csr->operation, alpha, csr->A, csr->descr, csr->layout, x, k, k, beta, y, k);
	#elif DOUBLE == 1
		mkl_sparse_d_mm(csr->operation, alpha, csr->A, csr->descr, csr->layout, x, k, k, beta, y, k);
	#endif

	if (csr->y == NULL)
	{
		csr->y = y;
	}
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
