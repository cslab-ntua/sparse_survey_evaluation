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

#ifdef SPMM_KERNEL
	#include "spmm/aspt_spmm.h"
#endif
#ifdef SDDMM_KERNEL
	#include "sddmm/aspt_sddmm.h"
#endif

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	int nr = 0, special_p = 0;
	double vari = 0;
	int * ja_aspt, * special, * special2, * mcsr_e, * mcsr_cnt;
	ValueType * a_aspt;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k ) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		nr = CEIL(m,BH)*BH;
		int npanel = CEIL(nr,BH);
		double avg = 0;

		special = (int *)malloc(sizeof(int)*nnz);
		special2 = (int *)malloc(sizeof(int)*nnz);
		mcsr_cnt = (int *)malloc(sizeof(int)*(npanel+1));
		int * mcsr_chk = (int *)malloc(sizeof(int)*(npanel+1));
		mcsr_e = (int *)malloc(sizeof(int)*nnz); // reduced later

		int * row_ptr0;
		int * col_idx0;
		ValueType * val0;

		row_ptr0 = (int *)malloc(sizeof(int)*(nr+1));
		memset(row_ptr0, 0, sizeof(int)*(nr+1));
		memcpy(row_ptr0, ia, sizeof(int)*(m+1));
		for(int i=m; i<nr; i++)
			row_ptr0[i+1] = row_ptr0[i];

		col_idx0 = (int *)malloc(sizeof(int)*nnz+256);
		memset(col_idx0, 0, sizeof(int)*nnz+256);
		memcpy(col_idx0, ja, sizeof(int)*nnz);
		val0 = (ValueType *)malloc(sizeof(ValueType)*nnz+256);
		memset(val0, 0, sizeof(ValueType)*nnz+256);
		memcpy(val0, a, sizeof(ValueType)*nnz);

		ja_aspt = (int *)malloc(sizeof(int)*nnz+256);
		a_aspt = (ValueType *)malloc(sizeof(ValueType)*nnz+256);

		aspt_preprocess_cpu(row_ptr0, col_idx0, val0, ja_aspt, a_aspt, n, nnz, nr, npanel, &avg, &vari, special, special2, &special_p, mcsr_e, mcsr_cnt, mcsr_chk);

		free(row_ptr0);
		free(col_idx0);
		free(val0);
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		free(ja_aspt);
		free(special);
		free(special2);
		free(mcsr_e);
		free(mcsr_cnt);
		free(a_aspt);
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
	csr->format_name = (char *) "ASpT-CPU";
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
		aspt_spmm_cpu(csr->ja_aspt, csr->a_aspt, x, y, k, csr->nr, csr->vari, csr->special, csr->special2, csr->special_p, csr->mcsr_e, csr->mcsr_cnt);
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
		aspt_sddmm_cpu(csr->ja_aspt, csr->a_aspt, x, y, out, k, csr->nr, csr->vari, csr->special, csr->special2, csr->special_p, csr->mcsr_e, csr->mcsr_cnt);
	#endif

	if (csr->out == NULL)
	{
		csr->out = out;
	}
}
