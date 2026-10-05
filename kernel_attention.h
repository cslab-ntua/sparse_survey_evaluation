#ifndef KERNEL_ATTENTION_H
#define KERNEL_ATTENTION_H

#include "macros/cpp_defines.h"

#include "bench_common.h"

// Mirrors 'Matrix_Format' from kernel.h, but for fused sparse attention instead of SpMM/SDDMM.
// Kept as a separate interface (instead of extending Matrix_Format) so that none of the existing
// SpMM/SDDMM kernel_*.cpp files are forced to implement an 'attention' method.
struct Matrix_Format_Attention
{
	char * format_name;
	long m;                         // num query rows    (sequence length N for the query side)
	long n;                         // num key/value rows (sequence length N for the key/value side)
	long nnz;                       // num allowed (query, key) pairs in the attention mask
	int d;                          // head dimension
	double mem_footprint;

	// Q: m x d, K: n x d, V: n x d, O: m x d (all dense, row-major).
	virtual void attention(ValueType * Q, ValueType * K, ValueType * V, ValueType * O, int d) = 0;

	Matrix_Format_Attention(long m, long n, long nnz, int d) : m(m), n(n), nnz(nnz), d(d)
	{
		mem_footprint = 0;
	}

	virtual ~Matrix_Format_Attention() {}
};

struct Matrix_Format_Attention * csr_to_format_attention(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int d);

#endif /* KERNEL_ATTENTION_H */
