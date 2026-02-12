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

#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_descr.h"
#include "aoclsparse.h"

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	aoclsparse_matrix A;
	aoclsparse_mat_descr descr; // aoclsparse_matrix_type_general
	aoclsparse_operation operation = aoclsparse_operation_none;
	aoclsparse_order order = aoclsparse_order_row;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		aoclsparse_index_base base = aoclsparse_index_base_zero;
		const int expected_calls = 128;

		aoclsparse_create_mat_descr(&descr);
		aoclsparse_set_mat_index_base(descr, base);
		#if DOUBLE == 0
			aoclsparse_create_scsr(&A, base, m, n, nnz, ia, ja, a);
		#elif DOUBLE == 1
			aoclsparse_create_dcsr(&A, base, m, n, nnz, ia, ja, a);
		#endif

		aoclsparse_set_mm_hint(A, operation, descr, expected_calls);
		aoclsparse_optimize(A);
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		aoclsparse_destroy_mat_descr(descr);
		aoclsparse_destroy(&A);
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
	csr->format_name = (char *) "AOCL";
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
		aoclsparse_scsrmm(csr->operation, alpha, csr->A, csr->descr, csr->order, x, k, k, beta, y, k);
	#elif DOUBLE == 1
		aoclsparse_dcsrmm(csr->operation, alpha, csr->A, csr->descr, csr->order, x, k, k, beta, y, k);
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
