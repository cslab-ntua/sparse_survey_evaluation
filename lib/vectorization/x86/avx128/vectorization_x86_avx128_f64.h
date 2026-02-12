#ifndef VECTORIZATION_X86_AVX128_F64_H
#define VECTORIZATION_X86_AVX128_F64_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef union __attribute__((packed, aligned(8))) { double  v;    double s[1]; int64_t si[1]; uint64_t su[1]; double sf[1]; }  vec_f64_1_t;
typedef union __attribute__((packed, aligned(8))) { __m128d v;    double s[2]; int64_t si[2]; uint64_t su[2]; double sf[2]; }  vec_f64_2_t;
typedef union __attribute__((packed, aligned(8))) { __m128d v[2]; double s[4]; int64_t si[4]; uint64_t su[4]; double sf[4]; }  vec_f64_4_t;
typedef union __attribute__((packed, aligned(8))) { __m128d v[4]; double s[8]; int64_t si[8]; uint64_t su[8]; double sf[8]; }  vec_f64_8_t;

#define vec_len_default_f64  2


#define vec_f64_2(val)                                     ( (vec_f64_2_t)  { .v = (typeof((vec_f64_2_t){0}.v)) val } )
#define vec_f64_1(val)                                     ( (vec_f64_1_t)  { .v = (typeof((vec_f64_1_t){0}.v)) val } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_array_f64_8(val)                               val.s
#define vec_array_f64_4(val)                               val.s
#define vec_array_f64_2(val)                               val.s
#define vec_array_f64_1(val)                               val.s

#define vec_set1_f64_8(val)                                vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_set1_f64_2(val);)
#define vec_set1_f64_4(val)                                vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_set1_f64_2(val);)
#define vec_set1_f64_2(val)                                vec_f64_2( _mm_set1_pd(val) )
#define vec_set1_f64_1(val)                                vec_f64_1( val )

#define vec_set_iter_f64_8(iter, expr)                     vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i].v = _mm_set_pd(vec_iter_expr_2(iter, 2*_i, expr));)
#define vec_set_iter_f64_4(iter, expr)                     vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i].v = _mm_set_pd(vec_iter_expr_2(iter, 2*_i, expr));)
#define vec_set_iter_f64_2(iter, expr)                     vec_f64_2( _mm_set_pd(vec_iter_expr_2(iter, 0, expr)) )
#define vec_set_iter_f64_1(iter, expr)                     vec_f64_1( vec_iter_expr_1(iter, 0, expr) )

#define vec_loadu_f64_8(ptr)                               vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_loadu_f64_2(((double *)(ptr)) + 2*_i);)
#define vec_loadu_f64_4(ptr)                               vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_loadu_f64_2(((double *)(ptr)) + 2*_i);)
#define vec_loadu_f64_2(ptr)                               vec_f64_2( _mm_loadu_pd((double *) ptr) )
#define vec_loadu_f64_1(ptr)                               vec_f64_1( (*((double *) ptr)) )

#define vec_storeu_f64_8(ptr, vec)                         vec_loop_stmt(4, _i, vec_storeu_f64_2(((double *)(ptr)) + 2*_i, vec.v[_i]);)
#define vec_storeu_f64_4(ptr, vec)                         vec_loop_stmt(2, _i, vec_storeu_f64_2(((double *)(ptr)) + 2*_i, vec.v[_i]);)
#define vec_storeu_f64_2(ptr, vec)                         _mm_storeu_pd((double *) ptr, vec.v)
#define vec_storeu_f64_1(ptr, vec)                         do { (*((double *) ptr)) = (vec.v); } while (0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_add_f64_2(a.v[_i], b.v[_i]);)
#define vec_add_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_add_f64_2(a.v[_i], b.v[_i]);)
#define vec_add_f64_2(a, b)                                vec_f64_2( _mm_add_pd(a.v, b.v) )
#define vec_add_f64_1(a, b)                                vec_f64_1( (a.v + b.v) )

#define vec_sub_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_sub_f64_2(a.v[_i], b.v[_i]);)
#define vec_sub_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_sub_f64_2(a.v[_i], b.v[_i]);)
#define vec_sub_f64_2(a, b)                                vec_f64_2( _mm_sub_pd(a.v, b.v) )
#define vec_sub_f64_1(a, b)                                vec_f64_1( (a.v - b.v) )

#define vec_mul_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_mul_f64_2(a.v[_i], b.v[_i]);)
#define vec_mul_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_mul_f64_2(a.v[_i], b.v[_i]);)
#define vec_mul_f64_2(a, b)                                vec_f64_2( _mm_mul_pd(a.v, b.v) )
#define vec_mul_f64_1(a, b)                                vec_f64_1( (a.v * b.v) )

#define vec_div_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_div_f64_2(a.v[_i], b.v[_i]);)
#define vec_div_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_div_f64_2(a.v[_i], b.v[_i]);)
#define vec_div_f64_2(a, b)                                vec_f64_2( _mm_div_pd(a.v, b.v) )
#define vec_div_f64_1(a, b)                                vec_f64_1( (a.v / b.v) )

#define vec_fmadd_f64_8(a, b, c)                           vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_fmadd_f64_2(a.v[_i], b.v[_i], c.v[_i]);)
#define vec_fmadd_f64_4(a, b, c)                           vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_fmadd_f64_2(a.v[_i], b.v[_i], c.v[_i]);)
#define vec_fmadd_f64_2(a, b, c)                           vec_f64_2( _mm_fmadd_pd(a.v, b.v, c.v) )
#define vec_fmadd_f64_1(a, b, c)                           vec_f64_1( ((a.v * b.v + c.v)) )


#define vec_reduce_add_f64_8(a)                            vec_reduce_expr(double, 4, 0, _tmp, _i, _tmp += vec_reduce_add_f64_2(a.v[_i]);)
#define vec_reduce_add_f64_4(a)                            vec_reduce_expr(double, 2, 0, _tmp, _i, _tmp += vec_reduce_add_f64_2(a.v[_i]);)

#define vec_reduce_add_f64_2(a)                                      \
({                                                                   \
	const __m128d low_64   = a.v;                                \
	const __m128d high_64  = _mm_unpackhi_pd(low_64, low_64);    \
	const __m128d x64      = _mm_add_sd(low_64, high_64);        \
	_mm_cvtsd_f64(x64);                                          \
})

#define vec_reduce_add_f64_1(a)                            (a.v)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

// #define vec_cmpgt_mask_f64_8(a, b)                      vec_loop_expr(vec_f64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_xor_f64_2(a.v[_i], b.v[_i]);)
// #define vec_cmpgt_mask_f64_4(a, b)                       vec_loop_expr(vec_f64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_xor_f64_2(a.v[_i], b.v[_i]);)
// #define vec_cmpgt_mask_f64_2(a, b)                       _mm_xor_pd(a, b)
// #define vec_cmpgt_mask_f64_1(a, b)                       (a ^ b)


#endif /* VECTORIZATION_X86_AVX128_F64_H */

