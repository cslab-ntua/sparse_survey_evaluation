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

#include "DTC-SpMM_kernel_v2.h"

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

	#ifndef METHOD
		#define METHOD 0 // Fallback plan
	#endif
	int method = METHOD;

	int num_nodes = 0, num_edges = 0, tc_count = 0, embedding_dim = 0;
	int num_row_windows,  blockSize_h,  blockSize_w;
	int * RowWindowOffset_ptr, * TCblockRowid_ptr, * TCblockoffset_ptr, * SparseAToXindex_ptr;
	uint8_t * TCblocktileId_ptr;
	int RowWindowOffset_size, TCblockRowid_size, TCblocktileId_size, TCblockoffset_size, SparseAToXindex_size;
	int * RowWindowOffset, * TCblockRowid, * TCblockoffset, * SparseAToXindex;
	uint8_t * TCblocktileId;
	
	dim3 grid;
	dim3 block;
	dim3 grid_split;
	dim3 block_split;
	dim3 grid_float4;
	dim3 block_float4;
	dim3 grid_float4_split;
	dim3 block_float4_split;
	dim3 grid_v2;
	dim3 block_v2;
	dim3 grid_float4_split_v2;
	dim3 block_float4_split_v2;

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
		gpuCudaErrorCheck(cudaStreamSynchronize(stream));

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		preprocess_gpu_wrapper(ia, ja,  m, n, nnz,
			&num_row_windows, &blockSize_h, &blockSize_w,
			&RowWindowOffset_ptr, &TCblockRowid_ptr, &TCblocktileId_ptr, &TCblockoffset_ptr, &SparseAToXindex_ptr,
			&RowWindowOffset_size, &TCblockRowid_size, &TCblocktileId_size, &TCblockoffset_size, &SparseAToXindex_size
		);

		gpuCudaErrorCheck(cudaMalloc((void**) &RowWindowOffset, RowWindowOffset_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &TCblockRowid, TCblockRowid_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &TCblocktileId, TCblocktileId_size * sizeof(uint8_t)));
		gpuCudaErrorCheck(cudaMalloc((void**) &TCblockoffset, TCblockoffset_size * sizeof(int)));
		gpuCudaErrorCheck(cudaMalloc((void**) &SparseAToXindex, SparseAToXindex_size * sizeof(int)));

	 	gpuCudaErrorCheck(cudaMemcpy(RowWindowOffset, RowWindowOffset_ptr, RowWindowOffset_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(TCblockRowid, TCblockRowid_ptr, TCblockRowid_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(TCblocktileId, TCblocktileId_ptr, TCblocktileId_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(TCblockoffset, TCblockoffset_ptr, TCblockoffset_size * sizeof(int), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(SparseAToXindex, SparseAToXindex_ptr, SparseAToXindex_size * sizeof(int), cudaMemcpyHostToDevice));

		num_nodes = m;
		if(method==5 || method==6)
			num_edges = TCblocktileId_size;
		else
			num_edges = nnz;
		tc_count = TCblockRowid_size;

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

		gpuCudaErrorCheck(cudaFree(RowWindowOffset));
		gpuCudaErrorCheck(cudaFree(TCblockRowid));
		gpuCudaErrorCheck(cudaFree(TCblocktileId));
		gpuCudaErrorCheck(cudaFree(TCblockoffset));
		gpuCudaErrorCheck(cudaFree(SparseAToXindex));

		free(RowWindowOffset_ptr);
		free(TCblockRowid_ptr);
		free(TCblocktileId_ptr);
		free(TCblockoffset_ptr);
		free(SparseAToXindex_ptr);
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
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "DTC-%d", METHOD);
	csr->format_name = strdup(buffer);
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
		// The rest of the preprocessing for DTC needs to happen here, because it includes "k", that can't be put in the Matrix_Format Struct...
		csr->embedding_dim = k;
		if (csr->method == 2) {
			if (csr->embedding_dim < 64)
				printf("This method requires k>=64. It will not run any GPU code.\n");
		} else if (csr->method == 3) {
			if (csr->embedding_dim < 32)
				printf("This method requires k>=32. It will not run any GPU code.\n");
		} else if (csr->method == 4) {
			if (csr->embedding_dim < 128)
				printf("This method requires k>=128. It will not run any GPU code.\n");
		} else if (csr->method == 6) {
			if (csr->embedding_dim < 128)
				printf("This method requires k>=128. It will not run any GPU code.\n");
		}

		const int WARPperBlock = csr->embedding_dim / csr->blockSize_h;
		const int WARPperBlock1 = csr->embedding_dim / 32;
		// v1 grid and block sizes
		csr->grid.x = csr->num_row_windows; csr->grid.y = 1; csr->grid.z = 1;
		csr->block.x = WARP_SIZE; csr->block.y = WARPperBlock; csr->block.z = 1;
		csr->grid_split.x = csr->num_row_windows; csr->grid_split.y = WARPperBlock / 4; csr->grid_split.z = 1;
		csr->block_split.x = WARP_SIZE; csr->block_split.y = 4; csr->block_split.z = 1;
		csr->grid_float4.x = csr->num_row_windows; csr->grid_float4.y = 1; csr->grid_float4.z = 1;
		csr->block_float4.x = WARP_SIZE; csr->block_float4.y = WARPperBlock1; csr->block_float4.z = 1;
		csr->grid_float4_split.x = csr->num_row_windows; csr->grid_float4_split.y = WARPperBlock1 / 4; csr->grid_float4_split.z = 1;
		csr->block_float4_split.x = WARP_SIZE; csr->block_float4_split.y = 4; csr->block_float4_split.z = 1;
		// v2 grid and block sizes
		csr->grid_v2.x = (csr->tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP; csr->grid_v2.y = 1; csr->grid_v2.z = 1;
		csr->block_v2.x = WARP_SIZE; csr->block_v2.y = WARPperBlock; csr->block_v2.z = 1;
		csr->grid_float4_split_v2.x = (csr->tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP; csr->grid_float4_split_v2.y = WARPperBlock1 / 4; csr->grid_float4_split_v2.z = 1;
		csr->block_float4_split_v2.x = WARP_SIZE; csr->block_float4_split_v2.y = 4; csr->block_float4_split_v2.z = 1;
	}

	if (csr->method == 0) {
		spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer<<<csr->grid, csr->block>>>(csr->RowWindowOffset, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d,  csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 1) {
		spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2<<<csr->grid, csr->block>>>(csr->RowWindowOffset, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d,  csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 2) {
		if (csr->embedding_dim >= 64)
			spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split<<<csr->grid_split, csr->block_split>>>(csr->RowWindowOffset, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d,  csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 3) {
		if (csr->embedding_dim >= 32)
			spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4<<<csr->grid_float4, csr->block_float4>>>(csr->RowWindowOffset, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d,  csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 4) {
		if (csr->embedding_dim >= 128)
			spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split<<<csr->grid_float4_split, csr->block_float4_split>>>(csr->RowWindowOffset, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d,  csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 5) {
		spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv<<<csr->grid_v2, csr->block_v2>>>(csr->TCblockRowid, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d, csr->tc_count, csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	} else if (csr->method == 6) {
		if (csr->embedding_dim >= 128)
			spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance<<<csr->grid_float4_split_v2, csr->block_float4_split_v2>>>(csr->TCblockRowid, csr->TCblocktileId, csr->TCblockoffset, csr->SparseAToXindex, csr->a_d, csr->tc_count, csr->num_nodes, csr->num_edges, csr->embedding_dim, csr->x_d, csr->y_d);
	}

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
