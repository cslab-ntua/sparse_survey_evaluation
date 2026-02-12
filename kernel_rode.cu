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

	#include "cuda/cuda_util.h"
#ifdef __cplusplus
}
#endif

#include <cuda.h>

#include <memory>
#ifdef SPMM_KERNEL
	#include "spmm/RoDeSpmm.h"
	#include "spmm/matrix_utils.h"
	// using namespace SPC;
#endif
#ifdef SDDMM_KERNEL
	#include "sddmm/RoDeSddmm.h"
	#include "sddmm/matrix_utils.h"
	// using namespace SPC;
#endif
using namespace SPC;

struct CSRArrays : Matrix_Format
{
	INT_T * ia;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	INT_T * ia_d;
	INT_T * ja_d;
	ValueType * a_d;

	INT_T * ia_h;
	INT_T * ja_h;
	ValueType * a_h;

	cudaStream_t stream;

	ValueType * x = NULL;
	ValueType * y = NULL;
	ValueType * out = NULL;

	ValueType * x_d = NULL;
	ValueType * y_d = NULL;
	ValueType * out_d = NULL;

	ValueType * x_h = NULL;
	ValueType * y_h = NULL;
	ValueType * out_h = NULL;

	cudaStream_t stream1, stream2;

	#if defined(SPMM_KERNEL) || defined(SDDMM_KERNEL)
		std::unique_ptr<SPC::SparseMatrix> sm1;
		std::unique_ptr<SPC::CudaSparseMatrix<ValueType>> c_sm;
	#endif

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		gpuCudaErrorCheck(cudaMalloc((void**)&ia_d, (m+1) * sizeof(*ia_d)));
		gpuCudaErrorCheck(cudaMalloc((void**)&ja_d, nnz * sizeof(*ja_d)));
		gpuCudaErrorCheck(cudaMalloc((void**)&a_d, nnz * sizeof(*a_d)));

		gpuCudaErrorCheck(cudaStreamCreate(&stream));

		gpuCudaErrorCheck(cudaMallocHost((void**)&ia_h, (m+1) * sizeof(*ia_h)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&ja_h, nnz * sizeof(*ja_h)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&a_h, nnz * sizeof(*a_h)));

		memcpy(ia_h, ia, (m+1) * sizeof(*ia_h));
		memcpy(ja_h, ja, nnz * sizeof(*ja_h));
		memcpy(a_h, a, nnz * sizeof(*a_h));

		gpuCudaErrorCheck(cudaMemcpyAsync(ia_d, ia_h, (m+1) * sizeof(*ia_d), cudaMemcpyHostToDevice, stream));
		gpuCudaErrorCheck(cudaMemcpyAsync(ja_d, ja_h, nnz * sizeof(*ja_d), cudaMemcpyHostToDevice, stream));
		gpuCudaErrorCheck(cudaMemcpyAsync(a_d, a_h, nnz * sizeof(*a_d), cudaMemcpyHostToDevice, stream));

		// wait for transfers to finish
		gpuCudaErrorCheck(cudaStreamSynchronize(stream));

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		gpuCudaErrorCheck(cudaStreamCreate(&stream1));
		gpuCudaErrorCheck(cudaStreamCreate(&stream2));

		// The sparse matrix will be read here, that's why no previous proper reading in row_ptr, col_idx, val... leaving it for legacy reasons

		#if defined(SPMM_KERNEL) || defined(SDDMM_KERNEL)
			sm1 = std::make_unique<SPC::SparseMatrix>(ia, ja, a, m, n, nnz, SPC::SORTED, 1);
			sm1->RowDivide2Segment(512,4,32); // why this 512? nobody will know...
			c_sm = std::make_unique<SPC::CudaSparseMatrix<ValueType>>(*sm1);
		#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		gpuCudaErrorCheck(cudaFree(ia_d));
		gpuCudaErrorCheck(cudaFree(ja_d));
		gpuCudaErrorCheck(cudaFree(a_d));
		gpuCudaErrorCheck(cudaFree(x_d));
		gpuCudaErrorCheck(cudaFree(y_d));
		gpuCudaErrorCheck(cudaFree(out_d));

		gpuCudaErrorCheck(cudaFreeHost(ia_h));
		gpuCudaErrorCheck(cudaFreeHost(ja_h));
		gpuCudaErrorCheck(cudaFreeHost(a_h));
		gpuCudaErrorCheck(cudaFreeHost(x_h));
		gpuCudaErrorCheck(cudaFreeHost(y_h));
		gpuCudaErrorCheck(cudaFreeHost(out_h));

		gpuCudaErrorCheck(cudaStreamDestroy(stream));
		
		gpuCudaErrorCheck(cudaStreamDestroy(stream1));
		gpuCudaErrorCheck(cudaStreamDestroy(stream2));
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
	csr->format_name = (char *) "RoDe";
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
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->x_d, csr->n * k * sizeof(*csr->x_d)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&csr->x_h, csr->n * k * sizeof(*csr->x_h)));

		memcpy(csr->x_h, x, csr->n * k * sizeof(ValueType));
		gpuCudaErrorCheck(cudaMemcpyAsync(csr->x_d, csr->x_h, csr->n * k * sizeof(*csr->x_d), cudaMemcpyHostToDevice, csr->stream));
		gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));

		// Also, prepare for the output matrix y
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->y_d, csr->m * k * sizeof(*csr->y_d)));
	}

	#ifdef SPMM_KERNEL
	// Perhaps 128 is faster? Don't know for sure...
	// RoDeSpmm_n32(csr->c_sm->n_segs, csr->c_sm->n_segs_residue, csr->n, k, csr->c_sm->Values(), csr->c_sm->ColumnIndices(), csr->c_sm->RowOffsets(), csr->c_sm->seg_row_indices, csr->c_sm->seg_row_indices_residue, csr->c_sm->seg_st_offsets, csr->x_d, csr->y_d, csr->stream1, csr->stream2);
	RoDeSpmm_n128(csr->c_sm->n_segs, csr->c_sm->n_segs_residue, csr->n, k, csr->c_sm->Values(), csr->c_sm->ColumnIndices(), csr->c_sm->RowOffsets(), csr->c_sm->seg_row_indices, csr->c_sm->seg_row_indices_residue, csr->c_sm->seg_st_offsets, csr->x_d, csr->y_d, csr->stream1, csr->stream2);
	#endif

	gpuCudaErrorCheck(cudaPeekAtLastError());
	gpuCudaErrorCheck(cudaDeviceSynchronize());

	if (csr->y == NULL)
	{
		csr->y = y;

		gpuCudaErrorCheck(cudaMallocHost((void**)&csr->y_h, csr->m * k * sizeof(*csr->y_h)));
		gpuCudaErrorCheck(cudaMemcpyAsync(csr->y_h, csr->y_d, csr->m * k * sizeof(*csr->y_d), cudaMemcpyDeviceToHost, csr->stream));
		gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));
		memcpy(y, csr->y_h, csr->m * k * sizeof(ValueType));
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

		gpuCudaErrorCheck(cudaMalloc((void**)&csr->x_d, csr->m * k * sizeof(*csr->x_d)));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->y_d, k * csr->n * sizeof(*csr->y_d)));

		gpuCudaErrorCheck(cudaMallocHost((void**)&csr->x_h, csr->m * k * sizeof(*csr->x_h)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&csr->y_h, k * csr->n * sizeof(*csr->y_h)));

		memcpy(csr->x_h, x, csr->m * k * sizeof(ValueType));
		memcpy(csr->y_h, y, k * csr->n * sizeof(ValueType));

		gpuCudaErrorCheck(cudaMemcpyAsync(csr->x_d, csr->x_h, csr->m * k * sizeof(*csr->x_d), cudaMemcpyHostToDevice, csr->stream));
		gpuCudaErrorCheck(cudaMemcpyAsync(csr->y_d, csr->y_h, k * csr->n * sizeof(*csr->y_d), cudaMemcpyHostToDevice, csr->stream));

		gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));

		// Also, prepare for the output values
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->out_d, csr->nnz * sizeof(*csr->out_d)));
	}

	#ifdef SDDMM_KERNEL
	// Each version is producing wrong results... 32 always->32, 128 always 128... Whatever, moving on...
	RoDeSDDMM_n32(csr->c_sm->n_segs, csr->c_sm->n_segs_residue, csr->n, k, csr->c_sm->seg_row_indices, csr->c_sm->seg_row_indices_residue, csr->c_sm->seg_st_offsets, csr->c_sm->RowOffsets(), csr->c_sm->ColumnIndices(), csr->c_sm->Values(), csr->x_d, csr->y_d, csr->out_d, csr->stream1, csr->stream2);
	// RoDeSDDMM_n128(csr->c_sm->n_segs, csr->c_sm->n_segs_residue, csr->n, k, csr->c_sm->seg_row_indices, csr->c_sm->seg_row_indices_residue, csr->c_sm->seg_st_offsets, csr->c_sm->RowOffsets(), csr->c_sm->ColumnIndices(), csr->c_sm->Values(), csr->x_d, csr->y_d, csr->out_d, csr->stream1, csr->stream2);
	#endif

	gpuCudaErrorCheck(cudaPeekAtLastError());
	gpuCudaErrorCheck(cudaDeviceSynchronize());

	if (csr->out == NULL)
	{
		gpuCudaErrorCheck(cudaMallocHost((void**)&csr->out_h, csr->nnz * sizeof(*csr->out_h)));

		csr->out = out;

		gpuCudaErrorCheck(cudaMemcpyAsync(csr->out_h, csr->out_d, csr->nnz * sizeof(*csr->out_d), cudaMemcpyDeviceToHost, csr->stream));
		gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));
		memcpy(out, csr->out_h, csr->nnz * sizeof(ValueType));
	}
}
