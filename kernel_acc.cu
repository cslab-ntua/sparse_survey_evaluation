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
#include "acc_spmm_v2.h"

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

	ValueType * x_h = NULL;
	ValueType * y_h = NULL;
	ValueType * out_h = NULL;

	bool load_balance = false;

	METCF<ValueType> metcf;
	METCFBit<ValueType> metcf_bit;
	BME<ValueType> bme;
	AdpBME<ValueType> adpbme;

	vint rowWndSize, tcLocalBitSize, dataSize, sparseA2BSize;
	vint groupOffsetSize, tcOffsetSize, rowIndicesSize;

	vint * ptr_group_offset, * ptr_tc_offset, * ptr_row_indices_offset;
	vint * ptr_adp_groupOffset, * ptr_adp_rowIndices;

	TCLOCAL_TYPE * ptr_tcLocalBit;
	MAT_VAL_TYPE * ptr_data;
	vint * ptr_sparseA2B;

	vint numBlocks = 0;
	vint denseX_size = 0;
	vint denseY_size = 0;

	vint * block2Idx_d, * data2Idx_d;
	TCLOCAL_TYPE * tcLocalBit_d;
	MAT_VAL_TYPE * dataA_d;
	MAT_PTR_TYPE * sparseA2B_d;
	MAT_PTR_TYPE * tc_offset_d;
	MAT_PTR_TYPE * adp_group_offset_d;
	MAT_PTR_TYPE * adp_row_indices_d;

	dim3 grid_size;
	dim3 block_size;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		// gpuCudaErrorCheck(cudaMalloc((void**)&ia_d, (m+1) * sizeof(*ia_d)));
		// gpuCudaErrorCheck(cudaMalloc((void**)&ja_d, nnz * sizeof(*ja_d)));
		// gpuCudaErrorCheck(cudaMalloc((void**)&a_d, nnz * sizeof(*a_d)));

		gpuCudaErrorCheck(cudaStreamCreate(&stream));

		// gpuCudaErrorCheck(cudaMallocHost((void**)&ia_h, (m+1) * sizeof(*ia_h)));
		// gpuCudaErrorCheck(cudaMallocHost((void**)&ja_h, nnz * sizeof(*ja_h)));
		// gpuCudaErrorCheck(cudaMallocHost((void**)&a_h, nnz * sizeof(*a_h)));

		// memcpy(ia_h, ia, (m+1) * sizeof(*ia_h));
		// memcpy(ja_h, ja, nnz * sizeof(*ja_h));
		// memcpy(a_h, a, nnz * sizeof(*a_h));

		// gpuCudaErrorCheck(cudaMemcpyAsync(ia_d, ia_h, (m+1) * sizeof(*ia_d), cudaMemcpyHostToDevice, stream));
		// gpuCudaErrorCheck(cudaMemcpyAsync(ja_d, ja_h, nnz * sizeof(*ja_d), cudaMemcpyHostToDevice, stream));
		// gpuCudaErrorCheck(cudaMemcpyAsync(a_d, a_h, nnz * sizeof(*a_d), cudaMemcpyHostToDevice, stream));

		// wait for transfers to finish
		// gpuCudaErrorCheck(cudaStreamSynchronize(stream));


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		COO<ValueType>* coo = (COO<ValueType>*)malloc(sizeof(COO<ValueType>));
		coo->rows = m;
		coo->cols = n;
		coo->nnz = nnz;
		coo->row = (vint*)malloc(nnz * sizeof(vint));
		coo->col = (vint*)malloc(nnz * sizeof(vint));
		coo->data = a;

		for(int i=0; i<m; i++){
			for(int j=ia[i]; j<ia[i+1]; j++){
				coo->row[j] = i;
				coo->col[j] = ja[j];
			}
		}
		// printf("i=0\t[ ");
		// int prev_row = 0;
		// for(int i=0; i<row_ptr[11]; i++){
		// 	if(coo->row[i] != prev_row){
		// 		printf("]\ni=%d\t[ ", coo->row[i]);
		// 		prev_row = coo->row[i];
		// 	}
		// 	printf("%lf ", coo->data[i]);
		// }

		CSR<ValueType> csr;
		csr.rows = m;
		csr.cols = n;
		csr.nnz = nnz;
		csr.row_ptr.assign(ia, ia + m + 1);
		csr.col_idx.assign(ja, ja + nnz);
		csr.data.assign(a, a + nnz);

		// for(int i=0; i<11; i++){
		// 	printf("i=%d\t[ ", i);
		// 	for(int j=csr.row_ptr[i]; j<csr.row_ptr[i+1]; j++){
		// 		printf("%lf ", csr.data[j]);
		// 	}
		// 	printf("]\n");
		// }

		metcf.CSR2METCF(csr);
		metcf_bit.METCF2METCFBit(metcf);
		// metcf_bit.printSize();
		// printf("tc num of METCF: %d\n", metcf_bit.rowWindowOffsetBit.size());

		bme.CSR2BME(csr);
		// printf("tc num of BME: %d\n", bme.tcLocalBit.size() * 64 / ROW_WINDOW / COL_WINDOW);

		adpbme.CSR2AdpBME(csr, 3, 1, 2);

		// printf("\n------------------------\n");
		// tf32_spmm from now on...

		rowWndSize = static_cast<vint>(metcf_bit.rowWindowOffsetBit.size());
		tcLocalBitSize = static_cast<vint>(metcf_bit.tcLocalBit.size());
		dataSize = static_cast<vint>(metcf_bit.data.size());
		sparseA2BSize = static_cast<vint>(metcf_bit.sparseA2B.size());

		groupOffsetSize = static_cast<vint>(bme.groupOffset.size());
		tcOffsetSize = static_cast<vint>(bme.tcOffset.size());
		rowIndicesSize	= static_cast<vint>(bme.rowIndices.size());

		ptr_tcLocalBit = (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
		ptr_data = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
		ptr_sparseA2B = (vint*)malloc(sizeof(vint) * sparseA2BSize);
		
		std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
		std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);
		std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);

		ptr_group_offset	   = (vint*)malloc(sizeof(vint) * groupOffsetSize);
		ptr_tc_offset		  = (vint*)malloc(sizeof(vint) * tcOffsetSize);
		ptr_row_indices_offset = (vint*)malloc(sizeof(vint) * rowIndicesSize);
		
		std::copy(bme.groupOffset.begin(), bme.groupOffset.end(), ptr_group_offset);
		std::copy(bme.tcOffset.begin(), bme.tcOffset.end(), ptr_tc_offset);
		std::copy(bme.rowIndices.begin(), bme.rowIndices.end(), ptr_row_indices_offset);

		ptr_adp_groupOffset   = (vint*)malloc(sizeof(vint) * static_cast<vint>(adpbme.groupOffset.size()));
		ptr_adp_rowIndices	= (vint*)malloc(sizeof(vint) * static_cast<vint>(adpbme.rowIndices.size()));

		std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_adp_groupOffset);
		std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_adp_rowIndices);

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

		// gpuCudaErrorCheck(cudaFreeHost(ia_h));
		// gpuCudaErrorCheck(cudaFreeHost(ja_h));
		// gpuCudaErrorCheck(cudaFreeHost(a_h));
		gpuCudaErrorCheck(cudaFreeHost(x_h));
		gpuCudaErrorCheck(cudaFreeHost(y_h));
		gpuCudaErrorCheck(cudaFreeHost(out_h));

		gpuCudaErrorCheck(cudaStreamDestroy(stream));

		gpuCudaErrorCheck(cudaFree(tcLocalBit_d));
		gpuCudaErrorCheck(cudaFree(dataA_d));
		gpuCudaErrorCheck(cudaFree(sparseA2B_d));
		gpuCudaErrorCheck(cudaFree(x_d));
		gpuCudaErrorCheck(cudaFree(y_d));
		gpuCudaErrorCheck(cudaFree(tc_offset_d));
		gpuCudaErrorCheck(cudaFree(adp_group_offset_d));
		gpuCudaErrorCheck(cudaFree(adp_row_indices_d));

		gpuCudaErrorCheck(cudaFree(block2Idx_d));
		gpuCudaErrorCheck(cudaFree(data2Idx_d));

		free(ptr_tcLocalBit);
		free(ptr_data);
		free(ptr_sparseA2B);
		free(ptr_adp_groupOffset);
		free(ptr_adp_rowIndices);

		free(ptr_group_offset);
		free(ptr_tc_offset);
		free(ptr_row_indices_offset);
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
	csr->format_name = (char *) "ACC";
	return csr;
}

//==========================================================================================================================================
//= Computation
//==========================================================================================================================================

void
compute_spmm(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y, int k)
{
	// const ValueType alpha = 1.0;
	// const ValueType beta = 0.0;
	if (csr->x == NULL)
	{
		csr->x = x;
		// gpuCudaErrorCheck(cudaMalloc((void**)&csr->x_d, csr->n * k * sizeof(*csr->x_d)));
		// gpuCudaErrorCheck(cudaMallocHost((void**)&csr->x_h, csr->n * k * sizeof(*csr->x_h)));

		// memcpy(csr->x_h, x, csr->n * k * sizeof(ValueType));
		// gpuCudaErrorCheck(cudaMemcpyAsync(csr->x_d, csr->x_h, csr->n * k * sizeof(*csr->x_d), cudaMemcpyHostToDevice, csr->stream));
		// gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));

		// // Also, prepare for the output matrix y
		// gpuCudaErrorCheck(cudaMalloc((void**)&csr->y_d, csr->m * k * sizeof(*csr->y_d)));

		/*------------------------------------------------------*/
		// The rest of the preprocessing for ACC needs to happen here, because it includes "k", that can't be put in the Matrix_Format Struct...

		csr->load_balance = false;

		/*------------------------------------------------------*/
		for(vint i = 0; i < csr->rowWndSize; ++i)
			if(csr->metcf_bit.rowWindowOffsetBit[i] == false) csr->numBlocks++;;

		csr->denseX_size = csr->n * k;
		csr->denseY_size = std::max(csr->numBlocks * ROW_WINDOW * k, (csr->adpbme.rowIndices.back() + 8) * k);

		MAT_VAL_TYPE * DenseMatX, * DenseMatY;
		DenseMatX = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * csr->denseX_size);
		DenseMatY = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * csr->denseY_size);

		init_vec1(csr->denseX_size, DenseMatX, 1.0);
		init_vec1(csr->numBlocks * ROW_WINDOW * k, DenseMatY, 0.0);

		vint* block2Idx = (vint*)malloc(sizeof(vint) * (csr->numBlocks + 1));
		memset(block2Idx, 0, sizeof(vint) * (csr->numBlocks + 1));
		vint b_idx = 0;
		vint cnt_TCblock = 0;
		vint AvgTCBlock = csr->rowWndSize / csr->numBlocks;
		vint sumWindow = 0;
		vint tmp_zero = 0;
		vint tmp_nonZero = 0;
		for(vint i = 0; i < csr->rowWndSize; ++i) {
			++cnt_TCblock;
			if(csr->metcf_bit.rowWindowOffsetBit[i] == false) {
				block2Idx[++b_idx] = (i + 1);
				if(cnt_TCblock - AvgTCBlock == 0) {
					++tmp_zero;
				} else {
					++tmp_nonZero;
				}
				sumWindow += std::abs(int(cnt_TCblock - AvgTCBlock));
				cnt_TCblock = 0;
			}
		}
		vint IBD = sumWindow / csr->numBlocks;

		if(IBD >= 8)
			csr->load_balance = true;
		else
			csr->load_balance = false;

		// std::cout << "load balance is : " << load_balance << " " << IBD << std::endl;
		if(b_idx != csr->numBlocks)
			std::cerr << "Error with handling rowWndOffsetBit" << std::endl;

		vint* block2IdxBME = (vint*)malloc(sizeof(vint) * (csr->numBlocks + 1));
		memset(block2IdxBME, 0, sizeof(vint) * (csr->numBlocks + 1));
		vint b_idxBME = 0;
		for(vint i = 0; i < csr->rowWndSize; ++i) {
			if(csr->metcf_bit.rowWindowOffsetBit[i] == false) {
				block2IdxBME[++b_idxBME] = (i + 1);
			}
		}
		if(b_idxBME != csr->numBlocks)
			std::cout << "Error with handling rowWndOffsetBit" << std::endl;
		// std::cout << " numBlocks is : " << csr->numBlocks << std::endl;

		/*------------------------------------------------------*/
		vint* data2Idx = (vint*)malloc(sizeof(vint) * (csr->rowWndSize + 1));
		memset(data2Idx, 0, sizeof(vint) * (csr->rowWndSize + 1));
		// vint data_bidx = 1;
		vint cnt_data_idx = 0; 
		vint d_i = 0;
		// printf("metcf_bit.tcLocalBit.size() = %ld, rowWndSize = %d\n", metcf_bit.tcLocalBit.size(), rowWndSize);
		for (vint i = 0; i < csr->metcf_bit.tcLocalBit.size(); ++i) {
			if(d_i < csr->rowWndSize){
				data2Idx[d_i++] = cnt_data_idx;
				cnt_data_idx += __builtin_popcountll(csr->metcf_bit.tcLocalBit[i]);
				// printf("%d: cnt_data_idx = %d (metcf_bit.tcLocalBit = %ld) \n", i, cnt_data_idx, metcf_bit.tcLocalBit[i]);
				// std::cout << i << " cnt_data_idx = " << cnt_data_idx << "(metcf_bit.tcLocalBit = " << metcf_bit.tcLocalBit[i] <<  " -> " << __builtin_popcountll(metcf_bit.tcLocalBit[i]) << ")" << std::endl;
			}
			else{
				// printf("d_i > rowWndSize\n");
				break;
			}
		}
		data2Idx[d_i] = cnt_data_idx;
		// std::cout << "metcf_bit.tcLocalBit.size() is : " << csr->metcf_bit.tcLocalBit.size() << std::endl;
		/*------------------------------------------------------*/

		gpuCudaErrorCheck(cudaMalloc((void**)&csr->data2Idx_d, sizeof(vint) * (csr->rowWndSize + 1)));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->block2Idx_d, sizeof(vint) * (csr->numBlocks + 1)));

		gpuCudaErrorCheck(cudaMalloc((void**)&csr->tcLocalBit_d, sizeof(TCLOCAL_TYPE) * csr->tcLocalBitSize));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->dataA_d, sizeof(MAT_VAL_TYPE) * csr->dataSize));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->sparseA2B_d, sizeof(MAT_PTR_TYPE) * csr->sparseA2BSize));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->x_d, sizeof(MAT_VAL_TYPE) * csr->denseX_size));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->y_d, sizeof(MAT_VAL_TYPE) * csr->denseY_size));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->tc_offset_d, sizeof(MAT_PTR_TYPE) * csr->tcOffsetSize));

		gpuCudaErrorCheck(cudaMalloc((void**)&csr->adp_group_offset_d, sizeof(MAT_PTR_TYPE) * csr->adpbme.groupOffset.size()));
		gpuCudaErrorCheck(cudaMalloc((void**)&csr->adp_row_indices_d, sizeof(MAT_PTR_TYPE) * csr->adpbme.rowIndices.size()));

		gpuCudaErrorCheck(cudaMemcpy(csr->block2Idx_d, block2Idx, sizeof(vint) * (csr->numBlocks + 1), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->data2Idx_d, data2Idx, sizeof(vint) * (csr->rowWndSize + 1), cudaMemcpyHostToDevice));

		gpuCudaErrorCheck(cudaMemcpy(csr->tcLocalBit_d, csr->ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * csr->tcLocalBitSize, cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->dataA_d, csr->ptr_data, sizeof(MAT_VAL_TYPE) * csr->dataSize, cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->sparseA2B_d, csr->ptr_sparseA2B, sizeof(MAT_PTR_TYPE) * csr->sparseA2BSize, cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->x_d, DenseMatX, sizeof(MAT_VAL_TYPE) * csr->denseX_size, cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->y_d, DenseMatY, sizeof(MAT_VAL_TYPE) * csr->denseY_size, cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->tc_offset_d, csr->ptr_tc_offset, sizeof(MAT_PTR_TYPE) * csr->tcOffsetSize, cudaMemcpyHostToDevice));

		gpuCudaErrorCheck(cudaMemcpy(csr->adp_group_offset_d, csr->ptr_adp_groupOffset, sizeof(MAT_PTR_TYPE) * csr->adpbme.groupOffset.size(), cudaMemcpyHostToDevice));
		gpuCudaErrorCheck(cudaMemcpy(csr->adp_row_indices_d, csr->ptr_adp_rowIndices, sizeof(MAT_PTR_TYPE) * csr->adpbme.rowIndices.size(), cudaMemcpyHostToDevice));

		if(!csr->load_balance) {
			int warpsPerBlk = 0;
			int threshold = 512;
			if(k <= threshold)
				warpsPerBlk = k / (COL_WINDOW_R << 1);
			else
				warpsPerBlk = k / (COL_WINDOW_R << 2);

			// dim3 grid_size(numBlocks, 1, 1);
			csr->grid_size.x = csr->numBlocks;
			csr->grid_size.y = 1;
			csr->grid_size.z = 1;
			// dim3 block_size(WARP_SIZE, warpsPerBlk, 1);
			csr->block_size.x = WARP_SIZE;
			csr->block_size.y = warpsPerBlk;
			csr->block_size.z = 1;
		} else {
			int warpsPerBlk = 0;
			int threshold = 128;
			if(k <= threshold)
				warpsPerBlk = k / (COL_WINDOW_R << 1);
			else
				warpsPerBlk = k / (COL_WINDOW_R << 2);

			// dim3 grid_size(adpbme.groupOffset.size()-1, 1, 1);
			csr->grid_size.x = csr->adpbme.groupOffset.size()-1;
			csr->grid_size.y = 1;
			csr->grid_size.z = 1;
			// dim3 block_size(WARP_SIZE, warpsPerBlk, 1);
			csr->block_size.x = WARP_SIZE;
			csr->block_size.y = warpsPerBlk;
			csr->block_size.z = 1;
		}
		/*------------------------------------------------------*/
		free(DenseMatX);
		free(DenseMatY);
		free(block2Idx);
		free(data2Idx);
		free(block2IdxBME);
	}

	if(!csr->load_balance)
		tf32TransposeCompute(csr->tcLocalBit_d, csr->sparseA2B_d, csr->dataA_d, csr->x_d, csr->y_d, csr->block2Idx_d, csr->data2Idx_d, csr->m, csr->numBlocks, k, csr->grid_size, csr->block_size);
	else
		tf32TransposeAdpBalanceCompute(csr->adp_group_offset_d, csr->tc_offset_d, csr->adp_row_indices_d, csr->tcLocalBit_d, csr->sparseA2B_d, csr->dataA_d, csr->x_d, csr->y_d, csr->adpbme, csr->numBlocks, csr->m, k, csr->denseY_size, csr->grid_size, csr->block_size);

	gpuCudaErrorCheck(cudaPeekAtLastError());
	gpuCudaErrorCheck(cudaDeviceSynchronize());

	if (csr->y == NULL)
	{
		csr->y = y;

		// gpuCudaErrorCheck(cudaMallocHost((void**)&csr->y_h, csr->m * k * sizeof(*csr->y_h)));
		// gpuCudaErrorCheck(cudaMemcpyAsync(csr->y_h, csr->y_d, csr->m * k * sizeof(*csr->y_d), cudaMemcpyDeviceToHost, csr->stream));
		// gpuCudaErrorCheck(cudaStreamSynchronize(csr->stream));
		// memcpy(y, csr->y_h, csr->m * k * sizeof(ValueType));

		gpuCudaErrorCheck(cudaMemcpy(csr->y, csr->y_d, sizeof(MAT_VAL_TYPE) * csr->denseY_size, cudaMemcpyDeviceToHost));
		memcpy(y, csr->y, csr->m * k * sizeof(ValueType));
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
