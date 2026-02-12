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

#ifdef SPMM_KERNEL
	#include "spmm/aspt_spmm_v2.h"
#endif
#ifdef SDDMM_KERNEL
	#include "sddmm/aspt_sddmm_v2.h"
#endif

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

	int nr = 0, npanel = 0, num_dense = 0, special_p = 0;
	double vari = 0;
	int d_flag[128];
	cudaStream_t stream1, stream2, stream3;

	int *_mcsr_cnt, *_mcsr_chk;
	int *_mcsr_e, *_mcsr_list;

	int *_csr_e; 
	ValueType *_csr_ev;
	int *_baddr, *_saddr;
	int *_special, *_special2;

	dim3 s_gridsize;
	dim3 s_blocksize;
	dim3 ss_gridsize;
	dim3 ss_blocksize;
	dim3 d_gridsize;
	dim3 d_blocksize;
	dim3 s_gridsizeh;
	dim3 s_blocksizeh;

	CSRArrays(INT_T * ia, INT_T * ja, ValueType * a, long m, long n, long nnz, int k) : Matrix_Format(m, n, nnz, k), ia(ia), ja(ja), a(a)
	{
		nr = CEIL(m,BH)*BH;
		npanel = CEIL(m,BH);

		// + 256 is required by the ASpT specification... Don't know why though.
		gpuCudaErrorCheck(cudaMalloc((void**)&ia_d, (nr+1) * sizeof(*ia_d)));
		gpuCudaErrorCheck(cudaMalloc((void**)&ja_d, nnz * sizeof(*ja_d) + 256));
		gpuCudaErrorCheck(cudaMalloc((void**)&a_d, nnz * sizeof(*a_d) + 256));

		gpuCudaErrorCheck(cudaMemset(ia_d, 0, (nr+1) * sizeof(*ia_d)));
		gpuCudaErrorCheck(cudaMemset(ja_d, 0, nnz * sizeof(*ja_d) + 256));
		gpuCudaErrorCheck(cudaMemset(a_d, 0, nnz * sizeof(*a_d) + 256));

		gpuCudaErrorCheck(cudaStreamCreate(&stream));

		gpuCudaErrorCheck(cudaMallocHost((void**)&ia_h, (nr+1) * sizeof(*ia_h)));
		gpuCudaErrorCheck(cudaMallocHost((void**)&ja_h, nnz * sizeof(*ja_h) + 256));
		gpuCudaErrorCheck(cudaMallocHost((void**)&a_h, nnz * sizeof(*a_h) + 256));

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
		gpuCudaErrorCheck(cudaStreamCreate(&stream3));

		gpuCudaErrorCheck(cudaMalloc((void **) &_mcsr_cnt, sizeof(int)*(npanel+1)));
		gpuCudaErrorCheck(cudaMalloc((void **) &_mcsr_chk, sizeof(int)*(npanel+1)));
		gpuCudaErrorCheck(cudaMemset(_mcsr_cnt, 0, sizeof(int)*(npanel+1)));
		gpuCudaErrorCheck(cudaMemset(_mcsr_chk, 0, sizeof(int)*(npanel+1)));

		gpuCudaErrorCheck(cudaMalloc((void **) &_mcsr_e, sizeof(int)*nnz+256*1));
		gpuCudaErrorCheck(cudaMalloc((void **) &_mcsr_list, sizeof(int)*nnz+256));
		gpuCudaErrorCheck(cudaMemset(_mcsr_e, 0, sizeof(int)*nnz+256*1));
		gpuCudaErrorCheck(cudaMemset(_mcsr_list, -1, sizeof(int)*nnz+256));

		int *_rmv;
		gpuCudaErrorCheck(cudaMalloc((void **) &_rmv, sizeof(int)*128));
		gpuCudaErrorCheck(cudaMemset(_rmv, 0, sizeof(int)*128));
		int *rmv = (int *)malloc(sizeof(int)*128);

		double *_vari;
		gpuCudaErrorCheck(cudaMalloc((void **) &_vari, sizeof(double)*128));
		gpuCudaErrorCheck(cudaMemset(_vari, 0, sizeof(double)*128));
		double *vari0 = (double *)malloc(sizeof(double)*128);

		int *_special_bb;
		gpuCudaErrorCheck(cudaMalloc((void **) &_special_bb, sizeof(int)*128));
		gpuCudaErrorCheck(cudaMemset(_special_bb, 0, sizeof(int)*128));
		int *special_bb = (int *)malloc(sizeof(int)*128);

		int *_scnt;
		gpuCudaErrorCheck(cudaMalloc((void **) &_scnt, sizeof(int)));
		gpuCudaErrorCheck(cudaMemset(_scnt, 0, sizeof(int)));

		int detect_nb = CEIL(nr, BH);
		int *_d_flag;
		gpuCudaErrorCheck(cudaMalloc((void **) &_d_flag, sizeof(int)*128));
		gpuCudaErrorCheck(cudaMemset(_d_flag, 0, sizeof(int)*128));

		int pivot_gen_nb = CEIL(npanel+1, 128);
		int *_csr_pivot;
		gpuCudaErrorCheck(cudaMalloc((void **) &_csr_pivot, sizeof(int)*(npanel+1)));
		gpuCudaErrorCheck(cudaMemset(_csr_pivot, 0, sizeof(int)*(npanel+1)));

		int pnt_gen_nb = CEIL(nnz+1, 128);
		int *_key; STYPE *_key2;
		int *_val;
		gpuCudaErrorCheck(cudaMalloc((void **) &_key, sizeof(int)*nnz+256));
		gpuCudaErrorCheck(cudaMalloc((void **) &_key2, sizeof(STYPE)*nnz+256));
		gpuCudaErrorCheck(cudaMalloc((void **) &_val, sizeof(int)*nnz+256));

		int fill_nb = CEIL(nnz, 128*4);
		int mcsr_nb = CEIL(nr, BH);
		int port_nb = CEIL(nnz, 128);

		// int mcsr_cnt_nb = CEIL(npanel, MCSR_CNT_TBSIZE);
		// int s_npanel = CEIL(npanel, 8);

		dense_block_detect<<<detect_nb, BH>>>(ia_d, _mcsr_chk, ja_d, _d_flag);
		gpuCudaErrorCheck(cudaMemcpy(d_flag, _d_flag, sizeof(int)*128, cudaMemcpyDeviceToHost));

		for(int i=1;i<128;i++) d_flag[0] += d_flag[i];

		double avg;

		if(d_flag[0] == 0) {
			num_dense = 0;
			avg = (double)nnz / nr;
			_mcsr_e = ia_d;
			_csr_e = ja_d;
			_csr_ev = a_d;
			simple_mcsr_cnt<<<pivot_gen_nb, 128>>>(npanel+1, _mcsr_cnt);
		}
		else {
			csr_pivot_gen<<<pivot_gen_nb, 128, 0, stream1>>>(npanel+1, ia_d, _csr_pivot);
			csr_pnt_gen<<<pnt_gen_nb, 128, 0, stream2>>>(nnz+1, ja_d, _key, _key2, _val);

			bb_segsort(_key, _val, nnz, _csr_pivot, npanel);

			mcsr_cnt_calc<<<npanel, MCSR_CNT_TBSIZE>>>(_csr_pivot, _key, _mcsr_cnt, _mcsr_chk);

			int *ttt=(int *)malloc(sizeof(int)*(npanel+1)); 	
			gpuCudaErrorCheck(cudaMemcpy(ttt, _mcsr_cnt, sizeof(int)*(npanel+1), cudaMemcpyDeviceToHost));
			for(int i=1;i<=npanel;i++) {
				ttt[i] += ttt[i-1]+1;
			}
			num_dense = ttt[npanel] - npanel;
			gpuCudaErrorCheck(cudaMemcpy(_mcsr_cnt, ttt, sizeof(int)*(npanel+1), cudaMemcpyHostToDevice));

			gpuCudaErrorCheck(cudaMalloc((void **) &_baddr, sizeof(int)*ttt[npanel]));
			gpuCudaErrorCheck(cudaMalloc((void **) &_saddr, sizeof(int)*ttt[npanel]));
			free(ttt);

			key2_marking<<<npanel, MCSR_CNT_TBSIZE>>>(_csr_pivot, _key, _key2, _val, _mcsr_cnt, _mcsr_list, _baddr, _saddr, _mcsr_chk);	

			fill_val<<<fill_nb, 128>>>(nnz, _val);

			bb_segsort(_key2, _val, nnz, ia_d, nr);
			
			fill_mcsre<<<mcsr_nb, BH>>>(ia_d, _mcsr_cnt, _key2, _mcsr_e, _rmv);

			//GPRINT(_mcsr_e, 100000);
			//exit(0);

			gpuCudaErrorCheck(cudaMemcpy(rmv, _rmv, sizeof(int)*128, cudaMemcpyDeviceToHost));
			gpuCudaErrorCheck(cudaMemcpy(&_mcsr_e[BH*(num_dense+npanel)], &nnz, sizeof(int), cudaMemcpyHostToDevice));
			for(int i=1;i<128;i++) 
				rmv[0] += rmv[i];
			avg = (double)rmv[0] / nr;

			// gpuCudaErrorCheck(cudaFree(_key));;
			// gpuCudaErrorCheck(cudaFree(_key2));;

			gpuCudaErrorCheck(cudaMalloc((void **) &_csr_e, sizeof(int)*nnz+256));
			gpuCudaErrorCheck(cudaMalloc((void **) &_csr_ev, sizeof(ValueType)*nnz+256));

			porting<<<port_nb, 128>>>(nnz, _val, ja_d, a_d, _csr_e, _csr_ev);
		}

		cal_vari<<<npanel, BH>>>(nr, avg, _mcsr_cnt, _mcsr_e, _vari, _special_bb);
		gpuCudaErrorCheck(cudaMemcpy(vari0, _vari, sizeof(double)*128, cudaMemcpyDeviceToHost));
		for(int i=1;i<128;i++)
			vari0[0] += vari0[i];
		vari = (double)vari0[0] / nr;

		// fprintf(stderr, "avg : %f, vari : %f, num_dense : %d\n", avg, vari, num_dense);

		if(vari >= 200) {
			gpuCudaErrorCheck(cudaMemcpy(special_bb, _special_bb, sizeof(int)*128, cudaMemcpyDeviceToHost));
			for(int i=0;i<128;i++) { 
				special_p += special_bb[i];
				//printf("%d\n", special_bb[i]);
			}
			gpuCudaErrorCheck(cudaMalloc((void **) &_special, sizeof(int)*special_p));
			gpuCudaErrorCheck(cudaMalloc((void **) &_special2, sizeof(int)*special_p));
			make_special<<<npanel, BH>>>(_mcsr_cnt, _mcsr_e, _special, _special2, _scnt);
		}

		cudaDeviceSynchronize();
		
		// Now free all the temporary variables
		gpuCudaErrorCheck(cudaFree(_rmv));
		gpuCudaErrorCheck(cudaFree(_vari));
		gpuCudaErrorCheck(cudaFree(_special_bb));
		gpuCudaErrorCheck(cudaFree(_scnt));
		gpuCudaErrorCheck(cudaFree(_d_flag));
		gpuCudaErrorCheck(cudaFree(_csr_pivot));
		gpuCudaErrorCheck(cudaFree(_key));
		gpuCudaErrorCheck(cudaFree(_key2));
		gpuCudaErrorCheck(cudaFree(_val));

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
		gpuCudaErrorCheck(cudaStreamDestroy(stream3));

		gpuCudaErrorCheck(cudaFree(_mcsr_cnt));
		gpuCudaErrorCheck(cudaFree(_mcsr_chk));
		gpuCudaErrorCheck(cudaFree(_mcsr_list));
		gpuCudaErrorCheck(cudaFree(_baddr));
		gpuCudaErrorCheck(cudaFree(_saddr));
		gpuCudaErrorCheck(cudaFree(_special));
		gpuCudaErrorCheck(cudaFree(_special2));

		if(d_flag[0] != 0){
			gpuCudaErrorCheck(cudaFree(_mcsr_e));
			gpuCudaErrorCheck(cudaFree(_csr_e));
			gpuCudaErrorCheck(cudaFree(_csr_ev));			
		}
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
	csr->format_name = (char *) "ASpT-GPU";
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
		// Now for the block and grid sizes
		csr->s_gridsize = dim3(csr->nr/SBF, 1, CEIL(k, MFACTOR*2));
		csr->s_blocksize = dim3(SBSIZE, 1, 1);
		csr->ss_gridsize = dim3(csr->nr/SBF, 1, CEIL(k, MFACTOR*2));
		csr->ss_blocksize = dim3(SBSIZE, 1, 1);
		csr->d_gridsize = dim3(csr->num_dense, 1, CEIL(k, MFACTOR*2));
		csr->d_blocksize = dim3(DBSIZE, 1, 1);
		csr->s_gridsizeh = dim3(csr->special_p, 1, CEIL(k, MFACTOR*2));
		csr->s_blocksizeh = dim3(SPBSIZE, 1, 1);
	}

	#ifdef SPMM_KERNEL
		if (csr->nnz/csr->n < 6 && csr->vari < 40) { 
			spmv_kernel64_ssparse<<<csr->ss_gridsize, csr->ss_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d);
		} else if (csr->vari < 200) {
			spmv_kernel64_sparse_v2<<<csr->s_gridsize, csr->s_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d);
			// need to check if the grid size is not zero, otherwise an error will occur later
			if(csr->d_gridsize.x != 0)
				spmv_kernel64_dense_v2<<<csr->d_gridsize, csr->d_blocksize, 0, csr->stream2>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->_baddr, csr->_saddr);
		} else {
			spmv_kernel64_sparse_v2l<<<csr->s_gridsize, csr->s_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d);
			if(csr->d_gridsize.x != 0)
				spmv_kernel64_dense_v2<<<csr->d_gridsize, csr->d_blocksize, 0, csr->stream2>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->_baddr, csr->_saddr);
			if(csr->s_gridsizeh.x !=0 )
				spmv_kernel64_sparse_v2h<<<csr->s_gridsizeh, csr->s_blocksizeh, 0, csr->stream3>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->_special, csr->_special2);
		}
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

		/*------------------------------------------------------*/
		// Now for the block and grid sizes
		csr->s_gridsize = dim3(csr->nr/SBF, 1, CEIL(k, MFACTOR*4));
		csr->s_blocksize = dim3(SBSIZE, 1, 1);
		csr->ss_gridsize = dim3(csr->nr/SBF, 1, CEIL(k, MFACTOR*2));
		csr->ss_blocksize = dim3(SBSIZE, 1, 1);
		csr->d_gridsize = dim3(csr->num_dense, 1, CEIL(k, MFACTOR*4));
		csr->d_blocksize = dim3(DBSIZE, 1, 1);
		csr->s_gridsizeh = dim3(csr->special_p, 1, CEIL(k, MFACTOR*4));
		csr->s_blocksizeh = dim3(SPBSIZE, 1, 1);
	}

	#ifdef SDDMM_KERNEL
		if (csr->nnz/csr->n < 6 && csr->vari < 40) { 
			spmv_kernel64_ssparse<<<csr->ss_gridsize, csr->ss_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->out_d);
		} else if (csr->vari < 200) {
			spmv_kernel128_sparse_v2<<<csr->s_gridsize, csr->s_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->out_d);
			// need to check if the grid size is not zero, otherwise an error will occur later
			if(csr->d_gridsize.x != 0)
				spmv_kernel128_dense_v2<<<csr->d_gridsize, csr->d_blocksize, 0, csr->stream2>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->_baddr, csr->_saddr, csr->out_d);
		} else {
			spmv_kernel128_sparse_v2l<<<csr->s_gridsize, csr->s_blocksize, 0, csr->stream1>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->out_d);
			if(csr->d_gridsize.x != 0)
				spmv_kernel128_dense_v2<<<csr->d_gridsize, csr->d_blocksize, 0, csr->stream2>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->_baddr, csr->_saddr, csr->out_d);
			if(csr->s_gridsizeh.x !=0 )
				spmv_kernel128_sparse_v2h<<<csr->s_gridsizeh, csr->s_blocksizeh, 0, csr->stream3>>>(k, csr->ia_d, csr->_csr_e, csr->_csr_ev, csr->_mcsr_cnt, csr->_mcsr_e, csr->_mcsr_list, csr->x_d, csr->y_d, csr->out_d, csr->_special, csr->_special2);
		}
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
