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

#include "HC-SpMM_kernel_v2.h"

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

	int num_row_windows = 0, num_nodes = 0, num_edges = 0;
	int blockSize_h = 0, blockSize_w = 0, embedding_dim = 0;
	int dynamic_shared_size = 0;
	
	dim3 grid;
	dim3 block;

	int * nodePointer_ptr, * edgeList_ptr, * blockPartition_ptr, * edgeToColumn_ptr, * edgeToRow_ptr, * hybrid_type_ptr, * row_nzr_ptr, * col_nzr_ptr;
	int nodePointer_size, edgeList_size, blockPartition_size, edgeToColumn_size, edgeToRow_size, hybrid_type_size, row_nzr_size, col_nzr_size;

	int * nodePointer, * edgeList, * blockPartition, * edgeToColumn, * edgeToRow, * hybrid_type, * row_nzr, * col_nzr;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		// gpuCudaErrorCheck(cudaMalloc((void**)&ia_d, (m+1) * sizeof(*ia_d)));
		// gpuCudaErrorCheck(cudaMalloc((void**)&ja_d, nnz * sizeof(*ja_d)));
		gpuCudaErrorCheck(cudaMalloc((void**)&a_d, nnz * sizeof(*a_d)));

		gpuCudaErrorCheck(cudaStreamCreate(&stream));

		// gpuCudaErrorCheck(cudaMallocHost((void**)&ia_h, (m+1) * sizeof(*ia_h)));
		// gpuCudaErrorCheck(cudaMallocHost((void**)&ja_h, nnz * sizeof(*ja_h)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&a_h, nnz * sizeof(*a_h)));

		// memcpy(ia_h, ia, (m+1) * sizeof(*ia_h));
		// memcpy(ja_h, ja, nnz * sizeof(*ja_h));
		memcpy(a_h, a, nnz * sizeof(*a_h));

		// gpuCudaErrorCheck(cudaMemcpyAsync(ia_d, ia_h, (m+1) * sizeof(*ia_d), cudaMemcpyHostToDevice, stream));
		// gpuCudaErrorCheck(cudaMemcpyAsync(ja_d, ja_h, nnz * sizeof(*ja_d), cudaMemcpyHostToDevice, stream));
		gpuCudaErrorCheck(cudaMemcpyAsync(a_d, a_h, nnz * sizeof(*a_d), cudaMemcpyHostToDevice, stream));

		// wait for transfers to finish
		// gpuCudaErrorCheck(cudaStreamSynchronize(stream));

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		preprocess_gpu_wrapper(ia, ja,  m, n, nnz,
			&num_row_windows, &blockSize_h, &blockSize_w,
			&nodePointer_ptr, &edgeList_ptr, &blockPartition_ptr, &edgeToColumn_ptr, &edgeToRow_ptr, &hybrid_type_ptr, &row_nzr_ptr, &col_nzr_ptr,
			&nodePointer_size, &edgeList_size, &blockPartition_size, &edgeToColumn_size, &edgeToRow_size, &hybrid_type_size, &row_nzr_size, &col_nzr_size);

		gpuCudaErrorCheck(cudaMalloc((void**) &nodePointer, nodePointer_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &edgeList, edgeList_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &blockPartition, blockPartition_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &edgeToColumn, edgeToColumn_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &edgeToRow, edgeToRow_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &hybrid_type, hybrid_type_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &row_nzr, row_nzr_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &col_nzr, col_nzr_size * sizeof(int)));

		gpuCudaErrorCheck(cudaMemcpy(nodePointer, nodePointer_ptr, nodePointer_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(edgeList, edgeList_ptr, edgeList_size * sizeof(int), cudaMemcpyHostToDevice));
	 	gpuCudaErrorCheck(cudaMemcpy(blockPartition, blockPartition_ptr, blockPartition_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(edgeToColumn, edgeToColumn_ptr, edgeToColumn_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(edgeToRow, edgeToRow_ptr, edgeToRow_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(hybrid_type, hybrid_type_ptr, hybrid_type_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(row_nzr, row_nzr_ptr, row_nzr_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(col_nzr, col_nzr_ptr, col_nzr_size * sizeof(int), cudaMemcpyHostToDevice));

		num_nodes = m;
		num_edges = nnz;

		grid.x = blockPartition_size;
		grid.y = 1;
		grid.z = 1;
		block.x = WARP_SIZE;
		block.y = WPB;
		block.z = 1;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	~CSRArrays()
	{
		free(a);
		free(ia);
		free(ja);

		// gpuCudaErrorCheck(cudaFree(ia_d));
		// gpuCudaErrorCheck(cudaFree(ja_d));
		gpuCudaErrorCheck(cudaFree(a_d));
		gpuCudaErrorCheck(cudaFree(x_d));
		gpuCudaErrorCheck(cudaFree(y_d));
		gpuCudaErrorCheck(cudaFree(out_d));

		// gpuCudaErrorCheck(cudaFreeHost(ia_h));
		// gpuCudaErrorCheck(cudaFreeHost(ja_h));
		// gpuCudaErrorCheck(cudaFreeHost(a_h));
		gpuCudaErrorCheck(cudaFreeHost(x_h));
		gpuCudaErrorCheck(cudaFreeHost(y_h));
		gpuCudaErrorCheck(cudaFreeHost(out_h));

		gpuCudaErrorCheck(cudaStreamDestroy(stream));

		free(nodePointer_ptr);
		free(edgeList_ptr);
		free(blockPartition_ptr);
		free(edgeToColumn_ptr);
		free(edgeToRow_ptr);
		free(hybrid_type_ptr);
		free(row_nzr_ptr);
		free(col_nzr_ptr);

		gpuCudaErrorCheck(cudaFree(nodePointer));
		gpuCudaErrorCheck(cudaFree(edgeList));
		gpuCudaErrorCheck(cudaFree(blockPartition));
		gpuCudaErrorCheck(cudaFree(edgeToColumn));
		gpuCudaErrorCheck(cudaFree(edgeToRow));
		gpuCudaErrorCheck(cudaFree(hybrid_type));
		gpuCudaErrorCheck(cudaFree(row_nzr));
		gpuCudaErrorCheck(cudaFree(col_nzr));

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
	csr->format_name = (char *) "HC";
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

		/*------------------------------------------------------*/
		// Now for the shared memory size
		csr->embedding_dim = k;

		int dimTileNum = (csr->embedding_dim + csr->blockSize_h - 1) / csr->blockSize_h;
		csr->dynamic_shared_size = dimTileNum * csr->blockSize_w * csr->blockSize_h * sizeof(ValueType); // dynamic shared memory.
	}

	spmm_forward_cuda_kernel_arbi_warps_hybrid_32<<<csr->grid, csr->block, csr->dynamic_shared_size>>>(csr->nodePointer, csr->edgeList, csr->blockPartition, csr->edgeToColumn, csr->edgeToRow, csr->a_d, csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d, csr->hybrid_type, csr->row_nzr, csr->col_nzr);

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
compute_sddmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, __attribute__((unused)) int k)
{
	__attribute__((unused)) const ValueType alpha = 1.0;
	__attribute__((unused)) const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
		csr->y = y;
	}

	gpuCudaErrorCheck(cudaPeekAtLastError());
	gpuCudaErrorCheck(cudaDeviceSynchronize());

	if (csr->out == NULL)
	{
		csr->out = out;
	}
}
