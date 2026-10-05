#include <stdio.h>
#include <stdlib.h>
#include <cmath>

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
#include "kernel_attention.h"

// Utils macro
#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))
#define Abs(x) ((x)>(0)?(x):-(x))

//
// Reference (gold) computation for the fused sparse attention kernel: for every query row i,
// runs a numerically-stable softmax restricted to the columns that are non-zero in the attention
// mask (row_ptr/col_idx), i.e. exactly the same set of (query, key) pairs the fused kernel visits,
// and compares the resulting weighted sum of V rows against the kernel output O.
//
double CheckAccuracy(INT_T * row_ptr, INT_T * col_idx, __attribute__((unused)) ValueType * mask_val,
    INT_T m, INT_T n, INT_T d,
    ValueType * Q, ValueType * K, ValueType * V, ValueType * O)
{
    __attribute__((unused)) ValueType epsilon_relaxed = 1e-4;
    #if DOUBLE == 0
        ValueType epsilon = 1e-3;
    #elif DOUBLE == 1
        ValueType epsilon = 1e-8;
    #endif

    (void) n;

    double scaling = 1.0 / sqrt((double) d);

    long max_degree = 0;
    for (long i = 0; i < m; i++)
        max_degree = Max(max_degree, (long) (row_ptr[i+1] - row_ptr[i]));

    double * scores = (double *) malloc((max_degree > 0 ? max_degree : 1) * sizeof(*scores));
    ValueType * o_gold = (ValueType *) malloc(d * sizeof(*o_gold));

    ValueType maxDiff = 0, diff;
    long i, j, c;

    for (i = 0; i < m; i++) {
        long j_s = row_ptr[i];
        long j_e = row_ptr[i+1];
        long degree = j_e - j_s;

        for (c = 0; c < d; c++)
            o_gold[c] = 0;

        if (degree > 0) {
            double max_score = -INFINITY;
            for (j = j_s; j < j_e; j++) {
                long key = col_idx[j];
                double dot = 0;
                for (c = 0; c < d; c++)
                    dot += (double) Q[i*d + c] * (double) K[key*d + c];
                double score = scaling * dot;
                scores[j - j_s] = score;
                if (score > max_score)
                    max_score = score;
            }

            double sum = 0, compensation = 0;
            for (j = j_s; j < j_e; j++) {
                double value, tmp;
                value = exp(scores[j - j_s] - max_score) - compensation;
                tmp = sum + value;
                compensation = (tmp - sum) - value;
                sum = tmp;
            }

            for (j = j_s; j < j_e; j++) {
                long key = col_idx[j];
                double p = exp(scores[j - j_s] - max_score) / sum;
                for (c = 0; c < d; c++)
                    o_gold[c] += (ValueType) (p * (double) V[key*d + c]);
            }
        }

        for (c = 0; c < d; c++) {
            diff = Abs(o_gold[c] - O[i*d + c]);
            if (Abs(o_gold[c]) > epsilon) {
                diff = diff / Abs(o_gold[c]);
                maxDiff = Max(maxDiff, diff);
            }
        }
    }

    if (maxDiff > epsilon)
        printf("Test failed! (%g)\n", (double) maxDiff);

    free(scores);
    free(o_gold);
    return (double) maxDiff;
}

int main(int argc, char **argv)
{
    if(argc<3){
        printf("Usage: %s <matrix_market_file> <d>\n", argv[0]);
        printf("  <matrix_market_file>: the sparsity pattern used as the attention mask (query x key)\n");
        printf("  <d>: head dimension of Q, K and V\n");
        exit(1);
    }

    int i = 1;
    double time_read, time_coo_to_csr, time_convert_to_format, time_compute;

    struct Matrix_Market * MTX = NULL;
    struct DLMC_Matrix * SMTX;
    ValueType * coo_val = NULL;
    INT_T * coo_rowind = NULL;
    INT_T * coo_colind = NULL;
    long coo_m = 0;
    long coo_n = 0;
    long coo_nnz = 0;

    ValueType * csr_a = NULL;
    INT_T * csr_ia = NULL;
    INT_T * csr_ja = NULL;
    long csr_m = 0;
    long csr_n = 0;
    long csr_nnz = 0;

    struct Matrix_Format_Attention * MF;

    ValueType * Q, * K, * V, * O;
    long iterations;

    char * file_in;
    file_in = argv[i++];
    char matrix_name[1000];
    snprintf(matrix_name, sizeof(matrix_name), "%s", file_in);

    int d = atoi(argv[i++]);
    char *dataset = getenv("DATASET");

    if (strcmp(dataset, "MATRIX_MARKET") == 0 || strcmp(dataset, "GRAPH") == 0 || strcmp(dataset, "MASKS") == 0) {
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

    if (csr_nnz==0)
    {
        printf("Error: matrix has no non-zeros\n");
        return 1;
    }
    if (csr_m != csr_n)
    {
        printf("Error: attention mask must be square (query and key sequence lengths must match), got %ld x %ld\n", csr_m, csr_n);
        return 1;
    }

    time_convert_to_format = time_it(1,
        MF = csr_to_format_attention(csr_ia, csr_ja, csr_a, csr_m, csr_n, csr_nnz, d);
    );

    unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
    Q = (typeof(Q)) aligned_alloc(64, csr_m * d * sizeof(*Q));
    #pragma omp parallel for
    for(long i=0; i<csr_m * d; ++i){
        Q[i] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    K = (typeof(K)) aligned_alloc(64, csr_n * d * sizeof(*K));
    #pragma omp parallel for
    for(long i=0; i<csr_n * d; ++i){
        K[i] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    V = (typeof(V)) aligned_alloc(64, csr_n * d * sizeof(*V));
    #pragma omp parallel for
    for(long i=0; i<csr_n * d; ++i){
        V[i] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    O = (typeof(O)) aligned_alloc(64, csr_m * d * sizeof(*O));
    #pragma omp parallel for
    for(long i=0; i<csr_m * d; i++)
        O[i] = 0.0;

    // warmup iteration
    MF->attention(Q, K, V, O, d);

    double check_acc = CheckAccuracy(csr_ia, csr_ja, csr_a, csr_m, csr_n, d, Q, K, V, O);

    if(check_acc < 0.1){
        const char* system = getenv("SYSTEM");
        if (system == NULL) {
            fprintf(stderr, "Environment variable SYSTEM not set.\n");
            exit(EXIT_FAILURE);
        }

        int gpu_kernel = 0;
        const char* env_gpu_kernel = getenv("GPU_KERNEL");
        if (env_gpu_kernel != NULL) {
            gpu_kernel = atoi(env_gpu_kernel);
        } else {
            fprintf(stderr, "Environment variable GPU_KERNEL not set.\n");
            exit(EXIT_FAILURE);
        }
        if(gpu_kernel)
            for(int i=0; i<1000; i++)
                MF->attention(Q, K, V, O, d);

        time_compute = 0;
        iterations = 128;
        for(int i=0; i<iterations; i++){
            time_compute += time_it(1,
                MF->attention(Q, K, V, O, d);
            );
        }
        // FLOP model: per non-zero (query, key) pair, the fused kernel performs a d-dim dot product
        // for the score (2*d flops, mirroring SDDMM's 2*nnz*k) plus a d-dim rescale-and-accumulate of
        // the output row (2*d flops, mirroring SpMM's 2*nnz*k) -> 4*d flops per non-zero in total.
        double gflops = 4.0 * MF->nnz * d * iterations / time_compute / 1e9;
        printf("Fused Attention kernel - matrix: %s (%ld rows, %ld cols, %ld nnz), read: %.4lf, coo_to_csr: %.4lf, format_conversion: %.4lf, format: %s, d: %d, system: %s, gflops: %.2lf\n", matrix_name, MF->m, MF->n, MF->nnz, time_read, time_coo_to_csr, time_convert_to_format, MF->format_name, d, system, gflops);
    }

    free(Q);
    free(K);
    free(V);
    free(O);

    free(csr_a);
    free(csr_ia);
    free(csr_ja);

    return 0;
}
