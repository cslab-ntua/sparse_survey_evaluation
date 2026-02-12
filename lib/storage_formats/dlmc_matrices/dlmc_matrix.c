#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <float.h>

#include "debug.h"
#include "genlib.h"
#include "io.h"
#include "string_util.h"
#include "parallel_io.h"
#include "plot/plot.h"

#include "dlmc_matrix.h"

#undef  TYPE
#undef  SUFFIX
#define TYPE  MATRIX_MARKET_FLOAT_T
#define SUFFIX  _f
#include "dlmc_matrix_gen.c"

#undef  TYPE
#undef  SUFFIX
#define TYPE  int
#define SUFFIX  _i
#include "dlmc_matrix_gen.c"

#undef  TYPE
#undef  SUFFIX
#define TYPE  complex MATRIX_MARKET_FLOAT_T
#define SUFFIX  _cf
#include "dlmc_matrix_gen.c"

#undef  TYPE
#undef  SUFFIX
#define TYPE  int
#define SUFFIX  _pat
#include "dlmc_matrix_gen.c"


/* 
 * Additional variants are defined for cases in which symmetries can be used to significantly reduce the size of the data: symmetric, skew-symmetric and Hermitian.
 * In these cases, only entries in the lower triangular portion need be supplied.
 * In the skew-symmetric case the diagonal entries are zero, and hence they too are omitted.
 */


#define cast_void_pointer_value(field, ptr)                                                                       \
(                                                                                                                 \
	(!strcmp(field, "real")) ?                                                                                \
		(MATRIX_MARKET_FLOAT_T *) ptr                                                                     \
	: (!strcmp(field, "integer")) ?                                                                           \
		(int *) ptr                                                                                       \
	: (!strcmp(field, "complex")) ?                                                                           \
		(complex MATRIX_MARKET_FLOAT_T *) ptr                                                             \
	: (!strcmp(field, "pattern")) ?                                                                           \
		NULL                                                                                              \
	: error("Error parsing MARKET matrix '%s': unrecognized field type: %s\n", MTX->filename, field), NULL    \
)


typeof(double (*) (void *, long))
smtx_functor_get_value(struct DLMC_Matrix * MTX)
{
	if (!strcmp(MTX->field, "real"))
	{
		MATRIX_MARKET_FLOAT_T * _tmp;
		return gen_functor_convert_basic_type_to_double(_tmp);
	}
	else if (!strcmp(MTX->field, "integer"))
	{
		int * _tmp;
		return gen_functor_convert_basic_type_to_double(_tmp);
	}
	else if (!strcmp(MTX->field, "complex"))
	{
		complex MATRIX_MARKET_FLOAT_T * _tmp;
		return gen_functor_convert_basic_type_to_double(_tmp);
	}
	else if (!strcmp(MTX->field, "pattern"))
		return NULL;
	else
		error("Error parsing MARKET matrix '%s': unrecognized field type: %s\n", MTX->filename, MTX->field);
	return NULL;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                      Constructor / Destructor                                                          -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void
smtx_init(struct DLMC_Matrix * MTX)
{
	MTX->filename = NULL;
	MTX->format = NULL;
	MTX->field = NULL;
	MTX->R = NULL;
	MTX->C = NULL;
	MTX->V = NULL;
}


struct DLMC_Matrix *
smtx_new()
{
	struct DLMC_Matrix * MTX;
	MTX = (typeof(MTX)) malloc(sizeof(*MTX));
	smtx_init(MTX);
	return MTX;
}


void
smtx_clean(struct DLMC_Matrix * MTX)
{
	if (MTX == NULL)
		return;
	free(MTX->filename);
	free(MTX->format);
	free(MTX->field);
	free(MTX->R);
	free(MTX->C);
	free(MTX->V);
}


void
smtx_destroy(struct DLMC_Matrix ** MTX_ptr)
{
	if (MTX_ptr == NULL)
		return;
	smtx_clean(*MTX_ptr);
	free(*MTX_ptr);
	*MTX_ptr = NULL;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                              Read File                                                                 -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


static inline
long
smtx_parse_header(struct File_Atoms * A, struct DLMC_Matrix * MTX)
{
	char ** lines;
	long i = 0, j;
	int num_chars;
	char * start, * object, * format, * field, * symmetry;
	char * buf = (char *) malloc(1000);
	int symmetric = 0, skew_symmetric = 0;

	lines = A->atoms;

	j = 0;
	// sscanf(lines[i] + j, "%s%n", buf, &num_chars);
	// j += num_chars;
	// start = strdup(buf);
	format = strdup("csr");
	field = strdup("real");
	symmetric = 0;
	skew_symmetric = 0;
	// if (strcmp(start, "%%MatrixMarket"))
	// {
	// 	// error("Error parsing MARKET matrix '%s': .mtx file did not start with %%MatrixMarket\n", MTX->filename);
	// 	format = strdup("coordinate");
	// 	field = strdup("real");
	// 	symmetric = 0;
	// 	skew_symmetric = 0;
	// }
	// else
	// {
	// 	sscanf(lines[i] + j, "%s%n", buf, &num_chars);
	// 	j += num_chars;
	// 	object = strdup(buf);
	// 	sscanf(lines[i] + j, "%s%n", buf, &num_chars);
	// 	j += num_chars;
	// 	format = strdup(buf);
	// 	if (strcmp(object, "matrix") || (strcmp(format, "coordinate") && strcmp(format, "array")))
	// 		error("Error parsing MARKET matrix '%s': only allow matrix coordinate or array format\n", MTX->filename, MTX->filename);

	// 	sscanf(lines[i] + j, "%s%n", buf, &num_chars);
	// 	j += num_chars;
	// 	field = strdup(buf);
	// 	sscanf(lines[i] + j, "%s%n", buf, &num_chars);
	// 	j += num_chars;
	// 	symmetry = strdup(buf);
	// 	i++;
	// 	// printf("%s - %s - %s - %s - %s\n", start.c_str(), object.c_str(), format.c_str(), field.c_str(), symmetry.c_str());

	// 	if (!strcmp(symmetry, "symmetric") || !strcmp(symmetry, "Hermitian"))
	// 	{
	// 		symmetric = 1;
	// 		skew_symmetric = 0;
	// 	}
	// 	else if (!strcmp(symmetry, "skew-symmetric"))
	// 	{
	// 		symmetric = 0;
	// 		skew_symmetric = 1;
	// 	}
	// 	else if (!strcmp(symmetry, "general"))
	// 	{
	// 		symmetric = 0;
	// 		skew_symmetric = 0;
	// 	}
	// 	else
	// 		error("Error parsing MARKET matrix '%s': unsupported symmetry type: %s\n", MTX->filename, symmetry);
	// }

	// while (lines[i][0] == '%')     // Comments.
	// 	i++;

	// M rows, N columns
	long M, K, nnz_sym, nnz;
	if (sscanf(lines[i], "%ld,%ld,%ld", &M, &K, &nnz_sym) != 3)
			error("Error parsing DLMC matrix '%s': invalid/missing matrix sizes: %s\n", MTX->filename, lines[i]);
	nnz = (symmetric || skew_symmetric) ? 2 * nnz_sym : nnz_sym;   // If symmetric, just place a worst case estimation for the mallocs.
	nnz_sym = nnz_sym;
	// if (!strcmp(format, "coordinate"))
	// {
	// 	if (sscanf(lines[i++], "%ld%ld%ld", &M, &N, &nnz_sym) != 3)
	// 		error("Error parsing MARKET matrix '%s': invalid/missing matrix sizes: %s\n", MTX->filename, lines[i-1]);
	// 	nnz = (symmetric || skew_symmetric) ? 2 * nnz_sym : nnz_sym;   // If symmetric, just place a worst case estimation for the mallocs.
	// 	nnz_sym = nnz_sym;
	// }
	// else
	// {
	// 	if (sscanf(lines[i++], "%ld%ld", &M, &N) != 2)
	// 		error("Error parsing MARKET matrix '%s': invalid/missing matrix sizes: %s\n", MTX->filename, lines[i-1]);
	// 	nnz = M*N;
	// 	nnz_sym = M*N;
	// }

	// if (nnz_sym != A->num_atoms - i)
	// 	error("Error parsing MARKET matrix '%s': remaining number of file lines (%ld) don't match the number of non-zeros (%d)\n", MTX->filename, A->num_atoms - i, nnz_sym);

	MTX->format = format;
	MTX->field = field;
	MTX->symmetric = symmetric;
	MTX->skew_symmetric = skew_symmetric;
	MTX->m = M;
	MTX->k = K;
	MTX->nnz = nnz;
	MTX->nnz_sym = nnz_sym;

	return ++i;
}

struct DLMC_Matrix *
smtx_read(char * filename, long expand_symmetry, long pattern_dummy_vals)
{
	struct DLMC_Matrix * MTX = malloc(sizeof(*MTX));
	struct File_Atoms * A;
	char ** lines;
	long * lengths;
	long i;
	A = (typeof(A)) malloc(sizeof(*A));
	file_to_lines(A, filename, 0);
	MTX->filename = strdup(filename);
	i = smtx_parse_header(A, MTX);
	// printf("line:%ld\n",i);
	if (!strcmp(MTX->format, "csr"))
	{
		MTX->R = malloc((MTX->m +1) * sizeof(*(MTX->R)));
		MTX->C = malloc(MTX->nnz * sizeof(*(MTX->C)));
	}
	else
	{
		MTX->R = NULL;
		MTX->C = NULL;
	}
	lines = &A->atoms[i];
	lengths = &A->atom_len[i];
	if (!strcmp(MTX->field, "real"))
	{
		#undef  SUFFIX
		#define SUFFIX  _f
		MTX->V = malloc(MTX->nnz * sizeof(MATRIX_MARKET_FLOAT_T));
		smtx_parse_data(lines, lengths, MTX, expand_symmetry);
	}
	// else if (!strcmp(MTX->field, "integer"))
	// {
	// 	#undef  SUFFIX
	// 	#define SUFFIX  _i
	// 	MTX->V = malloc(MTX->nnz * sizeof(int));
	// 	smtx_parse_data(lines, lengths, MTX, expand_symmetry);
	// }
	// else if (!strcmp(MTX->field, "complex"))
	// {
	// 	#undef  SUFFIX
	// 	#define SUFFIX  _cf
	// 	MTX->V = malloc(MTX->nnz * sizeof(complex MATRIX_MARKET_FLOAT_T));
	// 	smtx_parse_data(lines, lengths, MTX, expand_symmetry);
	// }
	// else if (!strcmp(MTX->field, "pattern"))
	// {
	// 	#undef  SUFFIX
	// 	#define SUFFIX  _pat
	// 	MTX->V = NULL;
	// 	smtx_parse_data(lines, lengths, MTX, expand_symmetry);
	// 	if (pattern_dummy_vals)
	// 	{
	// 		MTX->V = malloc(MTX->nnz * sizeof(MATRIX_MARKET_FLOAT_T));
	// 		#pragma omp parallel
	// 		{
	// 			#pragma omp for
	// 			for (i=0;i<MTX->nnz;i++)
	// 				((MATRIX_MARKET_FLOAT_T *) MTX->V)[i] = 1.0;
	// 		}
	// 	}
	// }
	else
		error("Error parsing MARKET matrix '%s': unrecognized field type: %s\n", MTX->filename, MTX->field);
	file_atoms_destroy(&A);
	return MTX;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                             Write File                                                                 -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// void
// smtx_write(struct DLMC_Matrix * MTX, char * filename)
// {
// 	char * field = MTX->field;
// 	char * str = NULL;
// 	long str_len = 0;
// 	if (!strcmp(field, "real"))
// 	{
// 		#undef  SUFFIX
// 		#define SUFFIX  _f
// 		str_len = smtx_to_string_par(MTX, &str);
// 	}
// 	// else if (!strcmp(field, "integer"))
// 	// {
// 	// 	#undef  SUFFIX
// 	// 	#define SUFFIX  _i
// 	// 	str_len = smtx_to_string_par(MTX, &str);
// 	// }
// 	// else if (!strcmp(field, "complex"))
// 	// {
// 	// 	#undef  SUFFIX
// 	// 	#define SUFFIX  _cf
// 	// 	str_len = smtx_to_string_par(MTX, &str);
// 	// }
// 	// else if (!strcmp(field, "pattern"))
// 	// {
// 	// 	#undef  SUFFIX
// 	// 	#define SUFFIX  _pat
// 	// 	str_len = smtx_to_string_par(MTX, &str);
// 	// }
// 	else
// 		error("Error parsing MARKET matrix '%s': unrecognized field type: %s\n", MTX->filename, field);
// 	int fd;
// 	fd = safe_open(filename, O_WRONLY | O_TRUNC | O_CREAT);
// 	safe_write(fd, str, str_len);
// 	// write_string_to_file(filename, str, str_len);

// }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                                Plot                                                                    -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// static
// void
// smtx_plot_base(struct DLMC_Matrix * MTX, char * filename, int plot_density)
// {
// 	struct Figure * fig;
// 	__attribute__((unused)) struct Figure_Series * s;
// 	long buf_n = strlen(filename) + 1 + 1000;
// 	char buf[buf_n];
// 	char * matrix_name;
// 	long len;

// 	// int num_pixels = 1024;
// 	int num_pixels = 2000;
// 	int num_pixels_x = num_pixels, num_pixels_y = num_pixels;
// 	// if (MTX->m < MTX->n)
// 		// num_pixels_x *= ((double) MTX->m) / ((double) MTX->n);
// 	// else if (MTX->m > MTX->n)
// 		// num_pixels_y *= ((double) MTX->n) / ((double) MTX->m);

// 	fig = malloc(sizeof(*fig));

// 	figure_init(fig, num_pixels_x, num_pixels_y);
// 	figure_axes_flip_y(fig);
// 	figure_set_bounds_x(fig, 0, MTX->k - 1);
// 	figure_set_bounds_y(fig, 0, MTX->m - 1);

// 	figure_enable_legend(fig);

// 	str_path_split_path(filename, strlen(filename)+1, buf, buf_n, NULL, &matrix_name);
// 	len = strlen(matrix_name);
// 	memmove(buf, matrix_name, len);

// 	snprintf(buf+len, buf_n-len, " ,  nnz=%ld", MTX->nnz);
// 	figure_set_title(fig, buf);

// 	if (!strcmp(MTX->format, "coordinate"))
// 		s = figure_add_series(fig, MTX->C, MTX->R, MTX->V, MTX->nnz, 0, , , smtx_functor_get_value(MTX));
// 	else
// 		s = figure_add_series(fig, NULL, NULL, MTX->V, MTX->k, MTX->m, , , smtx_functor_get_value(MTX));

// 	if (plot_density)
// 		figure_series_type_density_map(s);

// 	figure_plot(fig, filename);
// }


// void
// smtx_plot(struct DLMC_Matrix * MTX, char * filename)
// {
// 	smtx_plot_base(MTX, filename, 0);
// }


// void
// smtx_plot_density(struct DLMC_Matrix * MTX, char * filename)
// {
// 	smtx_plot_base(MTX, filename, 1);
// }

