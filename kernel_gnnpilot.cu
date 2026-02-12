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

#include "GNNPilot_v2.h"

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

	#ifndef BALANCE
		#define BALANCE 1 // Fallback plan
	#endif
	int balance = BALANCE, feat_len = 0, winfo_n = 0, feat_st = 0;
	int64_t ana_add;
	ana_info * ana;
	warp_info* winfo;

	const int kernel_len = 32;

	// for SPMM
	dim3 grid_naive;
	dim3 block_naive;
	dim3 grid_balance;
	dim3 block_balance;

	// for SDDMM
	dim3 grid;
	dim3 block;

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

		// no preprocessing on this side!

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

		kg_gcn_finalize(ana_add);
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
	snprintf(buffer, sizeof(buffer), "GNN-Pilot-%d", BALANCE);
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
		// The rest of the preprocessing for GNN-Pilot needs to happen here, because it includes "k", that can't be put in the Matrix_Format Struct...
		csr->feat_len = k;

		std::string op_name = "SpMM";
		csr->ana_add = get_ana(csr->ia, csr->ja, csr->m, csr->nnz, csr->feat_len, csr->balance, op_name);
		csr->ana = (ana_info*)csr->ana_add;
		csr->winfo = (warp_info*)csr->ana->winfo;
		csr->winfo_n = csr->ana->winfo_n;

		int thread_num_naive = csr->m * WARP_SIZE;
		int block_num_naive = (thread_num_naive + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int warp_num = (csr->winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
		int thread_num_balance = warp_num * WARP_SIZE;
		int block_num_balance = (thread_num_balance + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
		csr->grid_naive.x = block_num_naive;
		csr->block_naive.x = BLOCK_SIZE;
		
		csr->grid_balance.x = block_num_balance;
		csr->grid_balance.y = (csr->feat_len + csr->kernel_len - 1) / csr->kernel_len;
		
		csr->block_balance.x = BLOCK_SIZE_ALIGN;
	}

	if(csr->balance == 1){
		gcn_aggregate_kernel_naive<<<csr->grid_naive, csr->block_naive>>>(csr->m, csr->nnz, csr->feat_len, csr->ia_d, csr->ja_d, csr->a_d, csr->x_d, csr->y_d);
	}
	else{
		const int kernel_len = 32;
		gcn_aggregate_kernel_balance_aligned<kernel_len><<<csr->grid_balance, csr->block_balance>>>(csr->m, csr->nnz, csr->feat_len, csr->feat_st, csr->ia_d, csr->ja_d, csr->a_d, csr->x_d, csr->y_d, csr->winfo, csr->winfo_n);
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

		/*------------------------------------------------------*/
		// The rest of the preprocessing for GNN-Pilot needs to happen here, because it includes "k", that can't be put in the Matrix_Format Struct...
		csr->feat_len = k;

		std::string op_name = "SDDMM";
		csr->ana_add = get_ana(csr->ia, csr->ja, csr->m, csr->nnz, csr->feat_len, csr->balance, op_name);
		csr->ana = (ana_info*)csr->ana_add;
		csr->winfo = (warp_info*)csr->ana->winfo;
		csr->winfo_n = csr->ana->winfo_n;

		// neighbour grouping for balance
		int warp_num = (csr->winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
		int thread_num = warp_num * WARP_SIZE;
		int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

		csr->grid.x = block_num;
		csr->grid.y = (csr->feat_len + csr->kernel_len - 1) / csr->kernel_len;
		csr->block.x = BLOCK_SIZE_ALIGN;
	}

	const int kernel_len = 32;
	sddmm_aggregate_kernel_balance_aligned<kernel_len><<<csr->grid, csr->block>>>(csr->m, csr->nnz, csr->feat_len, csr->feat_st, csr->ia_d, csr->ja_d, csr->x_d, csr->y_d, csr->out_d, csr->winfo, csr->winfo_n);

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
