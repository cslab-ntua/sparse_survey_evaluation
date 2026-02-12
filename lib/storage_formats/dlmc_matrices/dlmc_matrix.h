#ifndef DLMC_MATRIX_H
#define DLMC_MATRIX_H

#include "macros/cpp_defines.h"
#include <complex.h>
#ifdef __cplusplus
	#define complex  _Complex
#endif



#ifndef MATRIX_MARKET_FLOAT_T
	#define MATRIX_MARKET_FLOAT_T  double
#endif


/*
 * Matrix in DLMC-CSR format.
 *
 * field: weight type -> real, integer, complex, pattern (none)
 *
 * m: num rows
 * k: num columns
 * nnz: num of non-zeros
 *
 * R: row indexes
 * C: column indexes
 * V: values
 */
struct DLMC_Matrix {
	char * filename;

	char * format;
	char * field;
	int symmetric;
	int skew_symmetric;

	long m;
	long k;
	long nnz;
	long nnz_sym;

	int * R;
	int * C;

	void * V;
};


void smtx_init(struct DLMC_Matrix * obj);
struct DLMC_Matrix * smtx_new();
void smtx_clean(struct DLMC_Matrix * obj);
void smtx_destroy(struct DLMC_Matrix ** obj_ptr);


double (* smtx_functor_get_value(struct DLMC_Matrix * MTX)) (void *, long);

struct DLMC_Matrix * smtx_read(char * filename, long expand_symmetry, long pattern_dummy_vals);
// void smtx_write(struct DLMC_Matrix * MTX, char * filename);
// void smtx_plot(struct DLMC_Matrix * MTX, char * filename);
// void smtx_plot_density(struct DLMC_Matrix * MTX, char * filename);


#endif /* DLMC_MATRIX_H */

