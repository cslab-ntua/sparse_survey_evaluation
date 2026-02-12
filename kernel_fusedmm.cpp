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

#define INDEXTYPE int64_t
#define VALUETYPE ValueType
#include "fusedMM.h"
extern "C" int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out);
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
	*out = val;;
	return FUSEDMM_SUCCESS_RETURN;
}

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	int64_t * row_ptr_64, * col_idx_64;
	int32_t imsg;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_ADD;
		col_idx_64 = (int64_t *)malloc(nnz * sizeof(int64_t));
		for (int i=0; i<nnz; i++)
			col_idx_64[i] = ja[i];
		row_ptr_64 = (int64_t *)malloc((m+1) * sizeof(int64_t));
		for (int i=0; i<m+1; i++)
			row_ptr_64[i] = ia[i];
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		free(row_ptr_64);
		free(col_idx_64);
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
	csr->format_name = (char *) "FusedMM";
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

	fusedMM_csr(csr->imsg, csr->m, csr->n, k, alpha, csr->nnz, csr->m, csr->n, csr->a, csr->col_idx_64, csr->row_ptr_64, csr->row_ptr_64 + 1, NULL, k, x, k, beta, y, k);

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
