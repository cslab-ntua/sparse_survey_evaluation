#include <stdio.h>
#include <stdlib.h>

#include "macros/cpp_defines.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "debug.h"
	#include "time_it.h"
	#include "string_util.h"
	#include "aux/csr_converter.h"
	#include "storage_formats/matrix_market/matrix_market.h"
	#include "storage_formats/dlmc_matrices/dlmc_matrix.h"
#ifdef __cplusplus
}
#endif

#include "bench_common.h"
#include "kernel.h"

// Utils macro
#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))
#define Abs(x) ((x)>(0)?(x):-(x))

double CheckAccuracy(INT_T * row_ptr, INT_T * col_idx, __attribute__((unused)) ValueType * val,
	INT_T csr_m, INT_T csr_n, INT_T csr_nnz, 
	INT_T dense_k,
	ValueType * x, ValueType * y, ValueType * out)
{
	__attribute__((unused)) ValueType epsilon_relaxed = 1e-4;
	#if DOUBLE == 0
		ValueType epsilon = 1e-5;
	#elif DOUBLE == 1
		ValueType epsilon = 1e-10;
	#endif
	// long i;
	ValueType * out_gold = (typeof(out_gold)) malloc(csr_nnz * sizeof(*out_gold));
	ValueType * out_test = (typeof(out_test)) malloc(csr_nnz * sizeof(*out_test));
	// #pragma omp parallel
	// {
		ValueType sum;
		long i, j;

		#pragma omp parallel for
		for(i=0;i<csr_nnz;i++)
		{
			out_gold[i] = 0;
			out_test[i] = out[i];
		}

		// #pragma omp for
		for (i = 0; i < csr_m; i++) {
			for (j = row_ptr[i]; j < row_ptr[i+1]; j++) {
				ValueType value, tmp, compensation;
				compensation = 0;
				sum = 0.0f;
				long curr_col = col_idx[j];
				for(long k = 0; k < dense_k; k++) {
					// value = val[j] * x[i*dense_k + k] * y[k*csr_n + curr_col] - compensation;
					// this would also be acceptable, since the values of sparse matrix are all set to 1 for SDDMM.
					value = x[i*dense_k + k] * y[k*csr_n + curr_col] - compensation;
					tmp = sum + value;
					compensation = (tmp - sum) - value;
					sum = tmp;
				}
				out_gold[j] = sum;
			}
		}
	// }

	ValueType maxDiff = 0, diff;
	for(i=0;i<csr_nnz;i++)
	{
		diff = Abs(out_gold[i] - out_test[i]);
		if (out_gold[i] > epsilon)
		{ 
			diff = diff / abs(out_gold[i]);
			maxDiff = Max(maxDiff, diff);
		}
	}
	if(maxDiff > epsilon)
		printf("Test failed! (%g)\n", (double)maxDiff);

	free(out_gold);
	free(out_test);
	return (double)maxDiff;
}

int main(int argc, char **argv)
{
	if(argc<3){
		printf("Usage: %s <matrix_market_file> <k>\n", argv[0]);
		exit(1);
	}

	int i = 1;
	double time_read, time_coo_to_csr, time_convert_to_format, time_compute;
	
	struct Matrix_Market * MTX = NULL;
	struct DLMC_Matrix * SMTX;
	ValueType * coo_val = NULL;   // MATRIX_MARKET_FLOAT_T is always double, as reference for error calculation.
	INT_T * coo_rowind = NULL;
	INT_T * coo_colind = NULL;
	long coo_m = 0;
	long coo_n = 0;
	long coo_nnz = 0;

	ValueType * csr_a = NULL;   // values (of size NNZ)
	INT_T * csr_ia = NULL;
	INT_T * csr_ja = NULL;
	long csr_m = 0;
	long csr_n = 0;
	long csr_nnz = 0;
	
	struct Matrix_Format * MF;   // Real matrices.

	ValueType * x, * y, * out;
	long iterations;

	char * file_in;
	file_in = argv[i++];
	char matrix_name[1000];
	snprintf(matrix_name, sizeof(matrix_name), "%s", file_in);

	int k = atoi(argv[i++]);
	char *dataset = getenv("DATASET");

	if (strcmp(dataset, "MATRIX_MARKET") == 0) {
		time_read = time_it(1,
			long expand_symmetry = 1;
			long pattern_dummy_vals = 1;
			MTX = mtx_read(file_in, expand_symmetry, pattern_dummy_vals);
			coo_rowind = MTX->R;
			coo_colind = MTX->C;
			coo_m = MTX->m;
			coo_n = MTX->n;
			coo_nnz = MTX->nnz;
			mtx_values_convert_to_real(MTX);
			coo_val = (typeof(coo_val)) MTX->V;
			MTX->R = NULL;
			MTX->C = NULL;
			MTX->V = NULL;
			mtx_destroy(&MTX);
		);
		// printf("time read: %lf\n", time_read);

		time_coo_to_csr = time_it(1,
			csr_a = (typeof(csr_a)) aligned_alloc(64, coo_nnz * sizeof(*csr_a));
			csr_ja = (typeof(csr_ja)) aligned_alloc(64, coo_nnz * sizeof(*csr_ja));
			csr_ia = (typeof(csr_ia)) aligned_alloc(64, (coo_m+1) * sizeof(*csr_ia));
			csr_m = coo_m;
			csr_n = coo_n;
			csr_nnz = coo_nnz;
			_Pragma("omp parallel for")
			for (long i=0;i<coo_nnz;i++)
			{
				csr_a[i] = 0.0;
				csr_ja[i] = 0;
			}
			_Pragma("omp parallel for")
			for (long i=0;i<coo_m+1;i++)
				csr_ia[i] = 0;
			coo_to_csr(coo_rowind, coo_colind, coo_val, coo_m, coo_n, coo_nnz, csr_ia, csr_ja, csr_a, 1, 0);

			free(coo_rowind);
			free(coo_colind);
			free(coo_val);
		);
	} else if (strcmp(dataset, "DLMC") == 0) {
		time_read = time_it(1,
			long expand_symmetry = 1;
			long pattern_dummy_vals = 1;
			SMTX = smtx_read(file_in, expand_symmetry, pattern_dummy_vals);
			coo_rowind = SMTX->R;
			coo_colind = SMTX->C;
			coo_m = SMTX->m;
			coo_n = SMTX->k;
			coo_nnz = SMTX->nnz;
			coo_val = (typeof(coo_val)) SMTX->V;
			SMTX->R = NULL;
			SMTX->C = NULL;
			SMTX->V = NULL;
			smtx_destroy(&SMTX);
		);

		time_coo_to_csr = time_it(1,
			csr_a = (typeof(csr_a)) aligned_alloc(64, coo_nnz * sizeof(*csr_a));
			csr_ja = (typeof(csr_ja)) aligned_alloc(64, coo_nnz * sizeof(*csr_ja));
			csr_ia = (typeof(csr_ia)) aligned_alloc(64, (coo_m+1) * sizeof(*csr_ia));
			csr_m = coo_m;
			csr_n = coo_n;
			csr_nnz = coo_nnz;
			_Pragma("omp parallel for")
			for (long i=0;i<coo_nnz;i++)
			{
				csr_a[i] = (ValueType) coo_val[i];
				csr_ja[i] = (long int) coo_colind[i];
			}
			_Pragma("omp parallel for")
			for (long i=0;i<coo_m+1;i++){
				csr_ia[i] = coo_rowind[i];
			}

			free(coo_rowind);
			free(coo_colind);
			free(coo_val);
		);

	} else {
		printf("Error: dataset not set\n");
		return 1;
	}
	// printf("time coo to csr: %lf\n", time_coo_to_csr);

	if(coo_nnz > 200)
	{
		_Pragma("omp parallel for")
		for (long i=0;i<coo_nnz;i++)
			csr_a[i] = 1.0;

		time_convert_to_format = time_it(1,
			MF = csr_to_format(csr_ia, csr_ja, csr_a, csr_m, csr_n, csr_nnz);
		);
		// printf("time convert to format: %lf\n", time_convert_to_format);

		x = (typeof(x)) aligned_alloc(64, csr_m * k * sizeof(*x));
		#pragma omp parallel for
		for(long i=0; i<csr_m * k; ++i)
			x[i] = 1.0;
		y = (typeof(y)) aligned_alloc(64, k * csr_n * sizeof(sizeof(*y)));
		#pragma omp parallel for
		for(long i=0; i<k * csr_n; i++)
			y[i] = 1.0;
		out = (typeof(out)) aligned_alloc(64, csr_nnz * sizeof(sizeof(*out)));
		#pragma omp parallel for
		for(long i=0; i<csr_nnz; i++)
			out[i] = 0.0;

		// warmup iteration
		MF->sddmm(x, y, out, k);

		// printf("---\nout = [ ");
		// for(int i=0; i<100; i++) printf("%lf ", out[i]);
		// printf("]\n---\n");
		double check_acc = CheckAccuracy(csr_ia, csr_ja, csr_a, csr_m, csr_n, csr_nnz, k, x, y, out);

		if(check_acc < 0.1){
			const char* system = getenv("SYSTEM");
			if (system == NULL) {
				// handle the case where the environment variable is not set
				fprintf(stderr, "Environment variable SYSTEM not set.\n");
				exit(EXIT_FAILURE);
			}

			// if GPU, need to run 1000 iterations more for warmup
			int gpu_kernel = 0;
			const char* env_gpu_kernel = getenv("GPU_KERNEL");
			if (env_gpu_kernel != NULL) {
				gpu_kernel = atoi(env_gpu_kernel);
			} else {
				// handle the case where the environment variable is not set
				fprintf(stderr, "Environment variable GPU_KERNEL not set.\n");
				exit(EXIT_FAILURE);
			}
			if(gpu_kernel)
				for(int i=0; i<1000; i++)
					MF->sddmm(x, y, out, k);

			time_compute = 0;
			iterations = 128;
			for(int i=0; i<iterations; i++){
				time_compute += time_it(1, 
					MF->sddmm(x, y, out, k);
				);
			}
			double gflops = 2.0 * MF->nnz * k * iterations / time_compute / 1e9;
			printf("SDDMM kernel - matrix: %s (%ld rows, %ld cols, %ld nnz), read: %.4lf, coo_to_csr: %.4lf, format_conversion: %.4lf, format: %s, k: %d, system: %s, gflops: %.2lf\n", matrix_name, MF->m, MF->n, MF->nnz, time_read, time_coo_to_csr, time_convert_to_format, MF->format_name, k, system, gflops);
		}		

		free(x);
		free(y);
		free(out);
	}

	free(csr_a);
	free(csr_ia);
	free(csr_ja);

	return 0;
}
