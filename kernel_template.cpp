#include <stdlib.h>
#include <stdio.h>

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

struct New_Array : Matrix_Format
{
	INT_T * row_ptr;      // the usual rowptr (of size m+1)
	INT_T * col_ind;      // the colidx of each NNZ (of size nnz)
	ValueType * values;   // the values (of size NNZ)

	New_Array(INT_T * row_ptr_in, INT_T * col_ind_in, ValueType * values_in, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k)
	{
		row_ptr = (typeof(row_ptr)) aligned_alloc(64, (m+1) * sizeof(*row_ptr));
		col_ind = (typeof(col_ind)) aligned_alloc(64, nnz * sizeof(*col_ind));
		values = (typeof(values)) aligned_alloc(64, nnz * sizeof(*values));
		#pragma omp parallel for
		for (long i=0;i<m+1;i++)
			row_ptr[i] = row_ptr_in[i];
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
		{
			values[i]=values_in[i];
			col_ind[i]=col_ind_in[i];
		}
	}

	~New_Array()
	{
		free(row_ptr);
		free(col_ind);
		free(values);
	}

	void spmm(ValueType * x, ValueType * y, int k);
	void sddmm(ValueType * x, ValueType * y, ValueType * out, int k) = 0;
};

void
New_Array::spmm(ValueType * x, ValueType * y, int k)
{

}

void
New_Array::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{

}

struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{

}
