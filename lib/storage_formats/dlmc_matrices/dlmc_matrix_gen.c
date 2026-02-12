#include "macros/macrolib.h"
#include "parallel_util.h"
#include <time.h>
// #include <random>

/* 
 * Additional variants are defined for cases in which symmetries can be used to significantly reduce the size of the data: symmetric, skew-symmetric and Hermitian.
 * In these cases, only entries in the lower triangular portion need be supplied.
 * In the skew-symmetric case the diagonal entries are zero, and hence they too are omitted.
 */

#undef  generic_name_expand
#define generic_name_expand(name)  CONCAT(name, SUFFIX)


#ifndef DLMC_MATRIX_GEN_C
#define DLMC_MATRIX_GEN_C

// #include "data_structures/vector/vector_gen_undef.h"
// #define VECTOR_GEN_TYPE_1  char
// #define VECTOR_GEN_SUFFIX  _mtx_c
// #include "data_structures/vector/vector_gen.c"

#endif /* DLMC_MATRIX_GEN_C */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                             Parse Data                                                                 -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#undef  generate_random_float
#define generate_random_float  generic_name_expand(generate_random_float)
MATRIX_MARKET_FLOAT_T generate_random_float(MATRIX_MARKET_FLOAT_T min, MATRIX_MARKET_FLOAT_T max) {
    return min + ((MATRIX_MARKET_FLOAT_T)rand() / (MATRIX_MARKET_FLOAT_T)RAND_MAX) * (max - min);
}
// void generate_random_numbers(MATRIX_MARKET_FLOAT_T* array, size_t size, MATRIX_MARKET_FLOAT_T min, MATRIX_MARKET_FLOAT_T max) {
//     _Pragma("omp parallel")
//     {
//         // Create a random number generator for each thread
//         std::random_device rd; // Seed generator
//         std::mt19937 generator(rd() + omp_get_thread_num()); // Mersenne Twister
//         std::uniform_real_distribution<MATRIX_MARKET_FLOAT_T> distribution(min, max);

//         _Pragma("omp for")
//         for (size_t i = 0; i < size; i++) {
//             array[i] = distribution(generator); // Generate random MATRIX_MARKET_FLOAT_T
//         }
//     }
// }

#undef  smtx_parse_array_format
#define smtx_parse_array_format  generic_name_expand(smtx_parse_array_format)
static
void
smtx_parse_array_format(char ** lines, long * lengths, struct DLMC_Matrix * MTX)
{
	// const int symmetric = MTX->symmetric || MTX->skew_symmetric;
	int * C =  MTX->C;
	int * R =  MTX->R;
	TYPE * V = (typeof(V)) MTX->V;
	// int complex_weights = (strcmp(MTX->field, "complex") == 0);
	long M, K, nnz, j;

	M = MTX->m;
	K = MTX->k;
	nnz = MTX->nnz;
	// _Pragma("omp parallel")
	// {

	// 	_Pragma("omp single")
	// 	{
	// 		_Pragma("omp task")
	// 		{
				char * ptr;
				ptr = lines[0];
				long k = 0;
				long len=0; //j = 0;
				long value;
				// printf("rows: %ld\n", M);

				for (j=0;j<M+1;j++)
				{
					len = gen_strtonum(ptr + k, lengths[0] - k, &R[j]);
					if (len==0){
						error("Error parsing MARKET matrix '%s': badly formed or missing value at matrix position (%ld)\n", MTX->filename, j);
					}
					// printf("%d %d ", R[j],MTX->R[j]);
					k += len;
				}
			// }
			// _Pragma("omp task")
			// {
				// char * ptr;
				ptr = lines[1];
				k = 0;
				// long len, j = 0;
				// printf("columns: %ld\n", K);
				for (j=0;j<nnz;j++)
				{
					len = gen_strtonum(ptr + k, lengths[1] - k, &C[j]);
					if (len == 0)
						error("Error parsing MARKET matrix '%s': badly formed or missing value at matrix position (%ld, %ld)\n", MTX->filename, 1, j);
					// printf("%d ", C[j]);
					k += len;
				}
			// }
	// 	}
	// }
		// generate_random_numbers(MTX->V,nnz,FLT_MIN,FLT_MAX);
	_Pragma("omp parallel")
	{
		_Pragma("omp for")
		for (j=0;j<nnz;j++)
		{
			// unsigned int seed = (unsigned int)(time(NULL) + j + omp_get_thread_num());
			// srand(seed);
			// V[j] = generate_random_float(-1.0,1.0);

			V[j]=1.0f;
		}
	}
}


// #undef  smtx_parse_coordinate_format
// #define smtx_parse_coordinate_format  generic_name_expand(smtx_parse_coordinate_format)
// static
// void
// smtx_parse_coordinate_format(char ** lines, long * lengths, struct DLMC_Matrix * MTX, long expand_symmetry)
// {
// 	int num_threads = safe_omp_get_num_threads_external();
// 	const int symmetric = MTX->symmetric || MTX->skew_symmetric;
// 	int * R = MTX->R;
// 	int * C = MTX->C;
// 	TYPE * V = (typeof(V)) MTX->V;
// 	int complex_weights = (strcmp(MTX->field, "complex") == 0);
// 	int non_diag_total = 0;
// 	int offsets[num_threads];
// 	int num_lines = MTX->nnz_sym;

// 	_Pragma("omp parallel")
// 	{
// 		int tnum = omp_get_thread_num();
// 		long i, i_s, i_e, j, len;
// 		char * ptr;
// 		long non_diag = 0;

// 		loop_partitioner_balance_iterations(num_threads, tnum, 0, num_lines, &i_s, &i_e);

// 		for (i=i_s;i<i_e;i++)
// 		{
// 			ptr = lines[i];
// 			j = 0;
// 			len = gen_strtonum(ptr + j, lengths[i] - j, &R[i]);
// 			// if (len == 0)
// 				// error("Error parsing MARKET matrix '%s': badly formed or missing row index at edge %ld\n", MTX->filename, i);
// 			j += len;
// 			R[i]--;   // From base-1 to base-0.
// 			len = gen_strtonum(ptr + j, lengths[i] - j, &C[i]);
// 			// if (len == 0)
// 				// error("Error parsing MARKET matrix '%s': badly formed or missing column index at edge %ld\n", MTX->filename, i);
// 			j += len;
// 			C[i]--;   // From base-1 to base-0.
// 			if (V != NULL)
// 			{
// 				len = gen_strtonum(ptr + j, lengths[i] - j, &V[i]);
// 				// if (len == 0)
// 					// error("Error parsing MARKET matrix '%s': badly formed or missing value at edge %ld\n", MTX->filename, i);
// 				j += len;
// 			}
// 			if (C[i] != R[i])
// 				non_diag++;
// 		}

// 		if (expand_symmetry && symmetric)
// 		{
// 			__atomic_fetch_add(&non_diag_total, non_diag, __ATOMIC_RELAXED);
// 			__atomic_store_n(&(offsets[tnum]), non_diag, __ATOMIC_RELAXED);

// 			_Pragma("omp barrier")

// 			_Pragma("omp single")
// 			{
// 				long a = 0, tmp;
// 				long diag = num_lines - non_diag_total;
// 				MTX->nnz = 2*non_diag_total + diag;
// 				for (i=0;i<num_threads;i++)
// 				{
// 					tmp = offsets[i];
// 					offsets[i] = a;
// 					a += tmp;
// 				}
// 			}

// 			_Pragma("omp barrier")

// 			long i;
// 			long j = num_lines + offsets[tnum];
// 			for (i=i_s;i<i_e;i++)
// 			{
// 				if (C[i] != R[i])
// 				{
// 					R[j] = C[i];
// 					C[j] = R[i];
// 					if (V != NULL)
// 					{
// 						if (complex_weights)
// 							V[j] = (MTX->skew_symmetric) ? -conj(V[i]) : conj(V[i]);
// 						else
// 							V[j] = (MTX->skew_symmetric) ? -V[i] : V[i];
// 					}
// 					j++;
// 				}
// 			}
// 		}
// 	}
// }


#undef  smtx_parse_data
#define smtx_parse_data  generic_name_expand(smtx_parse_data)
static
void
smtx_parse_data(char ** lines, long * lengths, struct DLMC_Matrix * MTX, long expand_symmetry)
{
	// if (!strcmp(MTX->format, "coordinate"))
	// 	smtx_parse_coordinate_format(lines, lengths, MTX, expand_symmetry);
	// else

	smtx_parse_array_format(lines, lengths, MTX);
	// printf("HI\n");
	// 	for (long j=0;j<MTX->m+1;j++)
	// {
	// 	printf("%d ",MTX->R[j]);
	// }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                          Convert to String                                                             -
//------------------------------------------------------------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


/*
#undef  smtx_to_string
#define smtx_to_string  generic_name_expand(smtx_to_string)
static
struct Vector *
smtx_to_string(struct DLMC_Matrix * MTX)
{
	int * R = MTX->R;
	int * C = MTX->C;
	TYPE * V = MTX->V;
	struct Vector * v;
	int buf_n = 10000, len;
	char buf[buf_n];
	long i;

	v = vector_new(0);

	len = snprintf(buf, buf_n, "%ld %ld %ld\n", MTX->m, MTX->k, MTX->nnz);
	vector_push_back_array(v, buf, len);

	for (i=0;i<MTX->nnz;i++)
	{
		len = 0;
		len += gen_numtostr(buf+len, buf_n-len, "", R[i] + 1);  // Base 1 arrays.
		buf[len++] = ' ';
		len += gen_numtostr(buf+len, buf_n-len, "", C[i] + 1);  // Base 1 arrays.
		if (strcmp(MTX->field, "pattern") != 0)
		{
			buf[len++] = ' ';
			len += gen_numtostr(buf+len, buf_n-len, "", V[i]);
		}
		len += snprintf(buf+len, buf_n-len, "\n");
		vector_push_back_array(v, buf, len);
	}
	return v;
} */


// Returns a 'page aligned' character array.

// #undef  smtx_to_string_par
// #define smtx_to_string_par  generic_name_expand(smtx_to_string_par)
// static
// long
// smtx_to_string_par(struct DLMC_Matrix * MTX, char ** str_ptr)
// {
// 	int num_threads = safe_omp_get_num_threads_external();
// 	int * R = MTX->R;
// 	int * C = MTX->C;
// 	TYPE * V = MTX->V;
// 	long offsets[num_threads];
// 	char * str;
// 	long str_len;

// 	#pragma omp parallel
// 	{
// 		int tnum = omp_get_thread_num();
// 		struct Vector * v;
// 		int buf_n = 10000, len;
// 		char buf[buf_n];
// 		long i, i_s, i_e;

// 		loop_partitioner_balance_iterations(num_threads, tnum, 0, MTX->nnz, &i_s, &i_e);

// 		v = vector_new(0);

// 		if (tnum == 0)
// 		{
// 			len = snprintf(buf, buf_n, "%ld %ld %ld\n", MTX->m, MTX->k, MTX->nnz);
// 			vector_push_back_array(v, buf, len);
// 		}

// 		for (i=i_s;i<i_e;i++)
// 		{
// 			len = 0;
// 			len += gen_numtostr(buf+len, buf_n-len, "", R[i] + 1);  // Base 1 arrays.
// 			buf[len++] = ' ';
// 			len += gen_numtostr(buf+len, buf_n-len, "", C[i] + 1);  // Base 1 arrays.
// 			if (strcmp(MTX->field, "pattern") != 0)
// 			{
// 				buf[len++] = ' ';
// 				len += gen_numtostr(buf+len, buf_n-len, "", V[i]);
// 			}
// 			len += snprintf(buf+len, buf_n-len, "\n");
// 			vector_push_back_array(v, buf, len);
// 		}

// 		offsets[tnum] = v->size;

// 		#pragma omp barrier
// 		#pragma omp single
// 		{
// 			long tmp;
// 			str_len = 0;
// 			for (i=0;i<num_threads;i++)
// 			{
// 				tmp = offsets[i];
// 				offsets[i] = str_len;
// 				str_len += tmp;
// 			}
// 			// str = malloc(str_len);
// 			str = aligned_alloc(sysconf(_SC_PAGESIZE), str_len);
// 		}
// 		#pragma omp barrier

// 		for (i=0;i<v->size;i++)
// 		{
// 			str[offsets[tnum] + i] = v->data[i];
// 		}

// 		vector_destroy(&v);
// 	}

// 	*str_ptr = str;
// 	return str_len;
// }

