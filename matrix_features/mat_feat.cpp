#include <stdio.h>
#include <stdlib.h>

// #include "read_coo_file.h"

#include "macros/cpp_defines.h"
// #include "read_mtx.h"
#include "../bench_common.h"
#include "../kernel.h"
#ifdef __cplusplus
extern "C"{
#endif
	#include "debug.h"
	#include "time_it.h"
	#include "string_util.h"
	#include "csr.h"
    #include "aux/csr_converter.h"
	#include "storage_formats/matrix_market/matrix_market.h"
	#include "storage_formats/dlmc_matrices/dlmc_matrix.h"
	#include "storage_formats/csr_util/csr_util_gen.h"
#ifdef __cplusplus
}
#endif




// #include "util.h"
// #include "matrix_util.h"


int main(int argc, char **argv)
{
	int n, m, nnz;
	ValueType * mtx_val;
	int * mtx_rowind;
	int * mtx_colind;

	struct Matrix_Market * MTX = NULL;
	struct DLMC_Matrix * SMTX;

	int * row_ptr;
	int * col_idx;
	ValueType * val;

	long buf_n = 1000;
	char buf[buf_n];
	double time;
	double time_read, time_coo_to_csr;
	long i;

	if (argc >= 6)
		return 1;

	char * file_in;
	char * path, * filename, * filename_base;


	i = 1;
	file_in = argv[i++];
    char *dataset = getenv("DATASET");

	str_path_split_path(file_in, strlen(file_in) + 1, buf, buf_n, &path, &filename);
	path = strdup(path);
	filename = strdup(filename);

	str_path_split_ext(filename, strlen(filename) + 1, buf, buf_n, &filename_base, NULL);
	filename_base = strdup(filename_base);
	snprintf(buf, buf_n, "figures/%s", filename_base);
	char * file_fig;
	file_fig = strdup(buf);

	printf("Matrix: %s\n", file_in);
    if (strcmp(dataset, "MATRIX_MARKET") == 0) {
		time_read = time_it(1,
			long expand_symmetry = 1;
			long pattern_dummy_vals = 1;
			MTX = mtx_read(file_in, expand_symmetry, pattern_dummy_vals);
			mtx_rowind = MTX->R;
			mtx_colind = MTX->C;
			m = MTX->m;
			n = MTX->n;
			nnz = MTX->nnz;
			row_ptr = (typeof(row_ptr)) malloc((m+1) * sizeof(*row_ptr));
			col_idx = (typeof(col_idx)) malloc(nnz * sizeof(*col_idx));
			val = (typeof(val)) malloc(nnz * sizeof(*val));
			mtx_values_convert_to_real(MTX);
			mtx_val = (typeof(mtx_val)) MTX->V;
			MTX->R = NULL;
			MTX->C = NULL;
			MTX->V = NULL;
			mtx_destroy(&MTX);
		);
		printf("time read: %lf\n", time_read);


		time_coo_to_csr = time_it(1,
			_Pragma("omp parallel for")
			for (long i=0;i<nnz;i++)
			{
				val[i] = 0.0;
				col_idx[i] = 0;
			}
			_Pragma("omp parallel for")
			for (long i=0;i<m+1;i++)
				row_ptr[i] = 0;
			printf("Converting COO to CSR...\n");
			coo_to_csr(mtx_rowind, mtx_colind, mtx_val, m, n, nnz, row_ptr, col_idx, val, 1, 0);
			printf("Conversion done.\n");
			free(mtx_rowind);
			free(mtx_colind);
			free(mtx_val);
		);
	} else if (strcmp(dataset, "DLMC") == 0) {
		printf("Reading DLMC matrix...\n");
		time_read = time_it(1,
			long expand_symmetry = 1;
			long pattern_dummy_vals = 1;
			SMTX = smtx_read(file_in, expand_symmetry, pattern_dummy_vals);
			row_ptr = (typeof(row_ptr)) malloc((m+1) * sizeof(*row_ptr));
			col_idx = (typeof(col_idx)) malloc(nnz * sizeof(*col_idx));
			val = (typeof(val)) malloc(nnz * sizeof(*val));
			row_ptr = SMTX->R;
			col_idx = SMTX->C;
			m = SMTX->m;
			n = SMTX->k;
			nnz = SMTX->nnz;
			val = (typeof(mtx_val)) SMTX->V;
			SMTX->R = NULL;
			SMTX->C = NULL;
			SMTX->V = NULL;
			smtx_destroy(&SMTX);

            free(mtx_rowind);
			free(mtx_colind);
			free(mtx_val);
		);

	} else {
		printf("Error: dataset not set\n");
		return 1;
	}
	printf("Matrix size: %d x %d with %d non-zeros\n", m, n, nnz);
    long num_pixels = 1024;
	long num_pixels_x = (n < num_pixels) ? n : num_pixels;
	long num_pixels_y = (m < num_pixels) ? m : num_pixels;
	if(m!=n) {
		double ratio = n*1.0 / m;
		if((ratio>16.0) || (ratio<(1/16.0)))
			ratio=16.0;
		// in order to keep both below 1024
		if(ratio>1) // n > m
			num_pixels_y = (1/ratio) * num_pixels_x;
		else // m > n
			num_pixels_x = ratio * num_pixels_y;
	}

	time = time_it(1,
		csr_plot_f(file_fig, row_ptr, col_idx, val, m, n, nnz, 0, num_pixels_x, num_pixels_y);
		csr_row_size_histogram_plot(file_fig, row_ptr, col_idx, val, m, n, nnz, 1, 1024, 1024);
		csr_num_neigh_histogram_plot(file_fig, row_ptr, col_idx, val, m, n, nnz, 1, 1, num_pixels_x, num_pixels_y);
		csr_cross_row_similarity_histogram_plot(file_fig, row_ptr, col_idx, val, m, n, nnz, 1, 1, num_pixels_x, num_pixels_y);
	);
	printf("time plot = %lf\n", time);

	csr_matrix_features_validation(filename_base, row_ptr, col_idx, m, n, nnz);


	free(row_ptr);
	free(col_idx);
	free(val);

	return 0;
}