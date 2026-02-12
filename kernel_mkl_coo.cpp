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
#ifdef __cplusplus
}
#endif

#include <mkl.h>


struct COOArrays : Matrix_Format
{
    INT_T * row_ind;      // holds the explicit row index of each NNZ (of size nnz)
    INT_T * col_ind;      // colidx of each NNZ (of size nnz)
    ValueType * val;   // the values (of size nnz)

    ValueType * x = NULL;
    ValueType * y = NULL;
    ValueType * out = NULL;

    sparse_matrix_t A;
    matrix_descr descr;
    const sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
    const sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;

    COOArrays(INT_T * row_ind, INT_T * col_ind, ValueType * val, long m, long n, long nnz, int k) 
        : Matrix_Format(m, n, nnz, k), row_ind(row_ind), col_ind(col_ind), val(val)
    {
        const sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
        const sparse_memory_usage_t policy = SPARSE_MEMORY_NONE;
        const int expected_calls = 128;

        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_verbose(1);


        #if DOUBLE == 0
            mkl_sparse_s_create_coo(&A, indexing, m, n, nnz, row_ind, col_ind, val);
        #elif DOUBLE == 1
            mkl_sparse_d_create_coo(&A, indexing, m, n, nnz, row_ind, col_ind, val);
        #endif

 
        mkl_sparse_set_mv_hint(A, operation, descr, expected_calls);
        mkl_sparse_set_memory_hint(A, policy);
        mkl_sparse_optimize(A);
    }

    ~COOArrays()
    {
        free(val);
        free(row_ind);
        free(col_ind);

        mkl_sparse_destroy(A);
    }

    void spmm(ValueType * x, ValueType * y, int k);
    void sddmm(ValueType * x, ValueType * y, ValueType * out, int k);
};

void compute_spmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k);
void compute_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, int k);

void
COOArrays::spmm(ValueType * x, ValueType * y, int k)
{
    compute_spmm(this, x, y, k);
}

void
COOArrays::sddmm(ValueType * x, ValueType * y, ValueType * out, int k)
{
    compute_sddmm(this, x, y, out, k);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz, int k)
{
    INT_T * row_ind = (INT_T *)malloc(nnz * sizeof(INT_T));
    if (row_ind == NULL) {
        fprintf(stderr, "Memory allocation failed for COO row indices.\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < m; ++i) {
        for (long j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            row_ind[j] = i;
        }
    }

    struct COOArrays * coo = new COOArrays(row_ind, col_ind, values, m, n, nnz, k);
    
    coo->mem_footprint = nnz * (sizeof(ValueType) + 2 * sizeof(INT_T));
    coo->format_name = (char *) "MKL_COO_IE"; 
    return coo;
}


void
compute_spmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, int k)
{
    const ValueType alpha = 1.0;
    const ValueType beta = 0.0;
    if (coo->x == NULL)
    {
        coo->x = x;
    }

    #if DOUBLE == 0
        mkl_sparse_s_mm(coo->operation, alpha, coo->A, coo->descr, coo->layout, x, k, k, beta, y, k);
    #elif DOUBLE == 1
        mkl_sparse_d_mm(coo->operation, alpha, coo->A, coo->descr, coo->layout, x, k, k, beta, y, k);
    #endif

    if (coo->y == NULL)
    {
        coo->y = y;
    }
}

void
compute_sddmm(COOArrays * restrict coo, ValueType * restrict x, ValueType * restrict y, ValueType * restrict out, __attribute__((unused)) int k)
{
    __attribute__((unused)) const ValueType alpha = 1.0;
    __attribute__((unused)) const ValueType beta = 0.0;
    if (coo->x == NULL)
    {
        coo->x = x;
        coo->y = y;
    }

    if (coo->out == NULL)
    {
        coo->out = out;
    }
}