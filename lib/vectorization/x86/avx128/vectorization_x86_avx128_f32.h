#ifndef VECTORIZATION_X86_AVX128_F32_H
#define VECTORIZATION_X86_AVX128_F32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef union __attribute__((packed, aligned(4))) { float  v;    float s[1];  int32_t si[1];  uint32_t su[1];  float sf[1];  }  vec_f32_1_t;
typedef union __attribute__((packed, aligned(4))) { __m128 v;    float s[4];  int32_t si[4];  uint32_t su[4];  float sf[4];  }  vec_f32_4_t;
typedef union __attribute__((packed, aligned(4))) { __m128 v[2]; float s[8];  int32_t si[8];  uint32_t su[8];  float sf[8];  }  vec_f32_8_t;
typedef union __attribute__((packed, aligned(4))) { __m128 v[4]; float s[16]; int32_t si[16]; uint32_t su[16]; float sf[16]; }  vec_f32_16_t;

#define vec_len_default_f32  4


#define vec_f32_4(val)                                     ( (vec_f32_4_t)  { .v = (typeof((vec_f32_4_t){0}.v)) val } )
#define vec_f32_1(val)                                     ( (vec_f32_1_t)  { .v = (typeof((vec_f32_1_t){0}.v)) val } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_array_f32_16(val)                              val.s
#define vec_array_f32_8(val)                               val.s
#define vec_array_f32_4(val)                               val.s
#define vec_array_f32_1(val)                               val.s

#define vec_set1_f32_16(val)                               vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_set1_f32_4(val);)
#define vec_set1_f32_8(val)                                vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_set1_f32_4(val);)
#define vec_set1_f32_4(val)                                vec_f32_4( _mm_set1_ps(val) )
#define vec_set1_f32_1(val)                                vec_f32_1( val )

#define vec_set_iter_f32_16(iter, expr)                    vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i].v = _mm_set_ps(vec_iter_expr_4(iter, 4*_i, expr));)
#define vec_set_iter_f32_8(iter, expr)                     vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i].v = _mm_set_ps(vec_iter_expr_4(iter, 4*_i, expr));)
#define vec_set_iter_f32_4(iter, expr)                     vec_f32_4( _mm_set_ps(vec_iter_expr_4(iter, 0, expr)) )
#define vec_set_iter_f32_1(iter, expr)                     vec_f32_1( vec_iter_expr_1(iter, 0, expr) )

#define vec_loadu_f32_16(ptr)                              vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_loadu_f32_4(((float *)(ptr)) + 4*_i);)
#define vec_loadu_f32_8(ptr)                               vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_loadu_f32_4(((float *)(ptr)) + 4*_i);)
#define vec_loadu_f32_4(ptr)                               vec_f32_4( _mm_loadu_ps((float *) ptr) )
#define vec_loadu_f32_1(ptr)                               vec_f32_1( (*((float *) ptr)) )

#define vec_storeu_f32_16(ptr, vec)                        vec_loop_stmt(4, _i, vec_storeu_f32_4(((float *)(ptr)) + 4*_i, vec.v[_i]);)
#define vec_storeu_f32_8(ptr, vec)                         vec_loop_stmt(2, _i, vec_storeu_f32_4(((float *)(ptr)) + 4*_i, vec.v[_i]);)
#define vec_storeu_f32_4(ptr, vec)                         _mm_storeu_ps((float *) ptr, vec.v)
#define vec_storeu_f32_1(ptr, vec)                         do { (*((float *) ptr)) = (vec.v); } while (0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_add_f32_4(a.v[_i], b.v[_i]);)
#define vec_add_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_add_f32_4(a.v[_i], b.v[_i]);)
#define vec_add_f32_4(a, b)                                vec_f32_4( _mm_add_ps(a.v, b.v) )
#define vec_add_f32_1(a, b)                                vec_f32_1( (a.v + b.v) )

#define vec_sub_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_sub_f32_4(a.v[_i], b.v[_i]);)
#define vec_sub_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_sub_f32_4(a.v[_i], b.v[_i]);)
#define vec_sub_f32_4(a, b)                                vec_f32_4( _mm_sub_ps(a.v, b.v) )
#define vec_sub_f32_1(a, b)                                vec_f32_1( (a.v - b.v) )

#define vec_mul_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_mul_f32_4(a.v[_i], b.v[_i]);)
#define vec_mul_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_mul_f32_4(a.v[_i], b.v[_i]);)
#define vec_mul_f32_4(a, b)                                vec_f32_4( _mm_mul_ps(a.v, b.v) )
#define vec_mul_f32_1(a, b)                                vec_f32_1( (a.v * b.v) )

#define vec_div_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_div_f32_4(a.v[_i], b.v[_i]);)
#define vec_div_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_div_f32_4(a.v[_i], b.v[_i]);)
#define vec_div_f32_4(a, b)                                vec_f32_4( _mm_div_ps(a.v, b.v) )
#define vec_div_f32_1(a, b)                                vec_f32_1( (a.v / b.v) )

#define vec_fmadd_f32_16(a, b, c)                          vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_fmadd_f32_4(a.v[_i], b.v[_i], c.v[_i]);)
#define vec_fmadd_f32_8(a, b, c)                           vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_fmadd_f32_4(a.v[_i], b.v[_i], c.v[_i]);)
#define vec_fmadd_f32_4(a, b, c)                           vec_f32_4( _mm_fmadd_ps(a.v, b.v, c.v) )
#define vec_fmadd_f32_1(a, b, c)                           vec_f32_1( ((a.v * b.v + c.v)) )


#define vec_reduce_add_f32_16(a)                           vec_reduce_expr(float, 4, 0, _tmp, _i, _tmp += vec_reduce_add_f32_4(a.v[_i]);)
#define vec_reduce_add_f32_8(a)                            vec_reduce_expr(float, 2, 0, _tmp, _i, _tmp += vec_reduce_add_f32_4(a.v[_i]);)

#define vec_reduce_add_f32_4(a)                                          \
({                                                                       \
	const __m128 low_64   = a.v;                                     \
	const __m128 high_64  = _mm_movehl_ps(low_64, low_64);           \
	const __m128 low_32   = _mm_add_ps(low_64, high_64);             \
	const __m128 high_32  = _mm_shuffle_ps(low_32, low_32, 0x01);    \
	const __m128 x32      = _mm_add_ss(low_32, high_32);             \
	_mm_cvtss_f32(x32);                                              \
})

#define vec_reduce_add_f32_1(a)                            (a.v)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

// #define vec_cmpgt_mask_f32_16(a, b)                      vec_loop_expr(vec_f32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_xor_f32_4(a.v[_i], b.v[_i]);)
// #define vec_cmpgt_mask_f32_8(a, b)                       vec_loop_expr(vec_f32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_xor_f32_4(a.v[_i], b.v[_i]);)
// #define vec_cmpgt_mask_f32_4(a, b)                       _mm_xor_ps(a, b)
// #define vec_cmpgt_mask_f32_1(a, b)                       (a ^ b)


#endif /* VECTORIZATION_X86_AVX128_F32_H */


