#ifndef VECTORIZATION_X86_AVX256_F32_H
#define VECTORIZATION_X86_AVX256_F32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"
#include "vectorization_x86_avx256_m32.h"


typedef union __attribute__((packed, aligned(4))) { float  v;    double  vf64;    int64_t vi;     float s[ 1]; int32_t si[ 1]; uint32_t su[ 1]; float sf[ 1]; }  vec_f32_1_t;
typedef union __attribute__((packed, aligned(4))) { __m128 v;    __m128d vf64;    __m128i vi;     float s[ 4]; int32_t si[ 4]; uint32_t su[ 4]; float sf[ 4]; }  vec_f32_4_t;
typedef union __attribute__((packed, aligned(4))) { __m256 v;    __m256d vf64;    __m256i vi;     float s[ 8]; int32_t si[ 8]; uint32_t su[ 8]; float sf[ 8]; }  vec_f32_8_t;
typedef union __attribute__((packed, aligned(4))) { __m256 v[2]; __m256d vf64[2]; __m256i vi[2];  float s[16]; int32_t si[16]; uint32_t su[16]; float sf[16]; }  vec_f32_16_t;
typedef union __attribute__((packed, aligned(4))) { __m256 v[4]; __m256d vf64[4]; __m256i vi[4];  float s[32]; int32_t si[32]; uint32_t su[32]; float sf[32]; }  vec_f32_32_t;

#define vec_len_default_f32     8
#define vec_len_default_f32_f2  16


#define vec_f32_8(val)                                     ( (vec_f32_8_t)  { .v = (val) } )
#define vec_f32_4(val)                                     ( (vec_f32_4_t)  { .v = (val) } )
#define vec_f32_1(val)                                     ( (vec_f32_1_t)  { .v = (val) } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Cast
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cast_to_i32_f32_32(val)                        vec_loop_expr(vec_i32_32_t, 4, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i32_f32_16(val)                        vec_loop_expr(vec_i32_16_t, 2, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i32_f32_8(val)                         ( (vec_i32_8_t)  { .v = (val).v } )
#define vec_cast_to_i32_f32_4(val)                         ( (vec_i32_4_t)  { .v = (val).v } )
#define vec_cast_to_i32_f32_1(val)                         ( (vec_i32_1_t)  { .v = (val).v } )

#define vec_cast_to_i64_f32_32(val)                        vec_loop_expr(vec_i64_16_t, 4, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i64_f32_16(val)                        vec_loop_expr(vec_i64_8_t,  2, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i64_f32_8(val)                         ( (vec_i64_4_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f32_4(val)                         ( (vec_i64_2_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f32_1(val)                         ( (vec_i64_1_t)  { .v = (val).vi } )

#define vec_cast_to_f32_f32_32(val)                        (val)
#define vec_cast_to_f32_f32_16(val)                        (val)
#define vec_cast_to_f32_f32_8(val)                         (val)
#define vec_cast_to_f32_f32_4(val)                         (val)
#define vec_cast_to_f32_f32_1(val)                         (val)

#define vec_cast_to_f64_f32_32(val)                        vec_loop_expr(vec_f64_16_t, 4, _tmp, _i, (_tmp).v[_i] = (val).vf64[_i];)
#define vec_cast_to_f64_f32_16(val)                        vec_loop_expr(vec_f64_8_t,  2, _tmp, _i, (_tmp).v[_i] = (val).vf64[_i];)
#define vec_cast_to_f64_f32_8(val)                         ( (vec_f64_4_t)  { .v = (val).vf64 } )
#define vec_cast_to_f64_f32_4(val)                         ( (vec_f64_2_t)  { .v = (val).vf64 } )
#define vec_cast_to_f64_f32_1(val)                         ( (vec_f64_1_t)  { .v = (val).vf64 } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_elem_get_f32_32(vec, index)                    ((vec).s[index])
#define vec_elem_get_f32_16(vec, index)                    ((vec).s[index])
#define vec_elem_get_f32_8(vec, index)                     ((vec).s[index])
#define vec_elem_get_f32_4(vec, index)                     ((vec).s[index])
#define vec_elem_get_f32_1(vec, index)                     ((vec).s[index])

#define vec_elem_set_f32_32(vec, index, expr)              { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f32_16(vec, index, expr)              { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f32_8(vec, index, expr)               { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f32_4(vec, index, expr)               { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f32_1(vec, index, expr)               { (vec).s[index] = (expr); } while (0)

#define vec_array_f32_32(vec)                              (vec).s
#define vec_array_f32_16(vec)                              (vec).s
#define vec_array_f32_8(vec)                               (vec).s
#define vec_array_f32_4(vec)                               (vec).s
#define vec_array_f32_1(vec)                               (vec).s

#define vec_set1_f32_32(val)                               vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_set1_ps(val);)
#define vec_set1_f32_16(val)                               vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_set1_ps(val);)
#define vec_set1_f32_8(val)                                vec_f32_8( _mm256_set1_ps(val) )
#define vec_set1_f32_4(val)                                vec_f32_4( _mm_set1_ps(val) )
#define vec_set1_f32_1(val)                                vec_f32_1( (float) (val) )

#define vec_set_iter_f32_32(iter, expr)                    vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_set_ps(vec_iter_expr_8(iter, 8*_i, (expr)));)
#define vec_set_iter_f32_16(iter, expr)                    vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_set_ps(vec_iter_expr_8(iter, 8*_i, (expr)));)
#define vec_set_iter_f32_8(iter, expr)                     vec_f32_8( _mm256_set_ps(vec_iter_expr_8(iter, 0, (expr))) )
#define vec_set_iter_f32_4(iter, expr)                     vec_f32_4( _mm_set_ps(vec_iter_expr_4(iter, 0, (expr))) )
#define vec_set_iter_f32_1(iter, expr)                     vec_f32_1( (float) vec_iter_expr_1(iter, 0, (expr)) )

#define vec_loadu_f32_32(ptr)                              vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_loadu_f32_16(ptr)                              vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_loadu_f32_8(ptr)                               vec_f32_8( _mm256_loadu_ps((float *) (ptr)) )
#define vec_loadu_f32_4(ptr)                               vec_f32_4( _mm_loadu_ps((float *) (ptr)) )
#define vec_loadu_f32_1(ptr)                               vec_f32_1( (*((float *) (ptr))) )

#define vec_loadu_maskedz_f32_32(ptr, mask)                vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_maskload_ps(((float *) (ptr)) + 8*_i, (mask).v[_i]);)
#define vec_loadu_maskedz_f32_16(ptr, mask)                vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_maskload_ps(((float *) (ptr)) + 8*_i, (mask).v[_i]);)
#define vec_loadu_maskedz_f32_8(ptr, mask)                 vec_f32_8( _mm256_maskload_ps((float *) (ptr), (mask).v) )
#define vec_loadu_maskedz_f32_4(ptr, mask)                 vec_f32_4( _mm_maskload_ps((float *) (ptr), (mask).v) )
#define vec_loadu_maskedz_f32_1(ptr, mask)                 vec_f32_1( ((mask).v) ? (*((float *) (ptr))) : 0 )

#define vec_storeu_f32_32(ptr, vec)                        vec_loop_stmt(4, _i, _mm256_storeu_ps(((float *) (ptr)) + 8*_i, (vec).v[_i]);)
#define vec_storeu_f32_16(ptr, vec)                        vec_loop_stmt(2, _i, _mm256_storeu_ps(((float *) (ptr)) + 8*_i, (vec).v[_i]);)
#define vec_storeu_f32_8(ptr, vec)                         _mm256_storeu_ps((float *) (ptr), (vec).v)
#define vec_storeu_f32_4(ptr, vec)                         _mm_storeu_ps((float *) (ptr), (vec).v)
#define vec_storeu_f32_1(ptr, vec)                         do { (*((float *) (ptr))) = ((vec).v); } while (0)

#define vec_storeu_masked_f32_32(ptr, vec, mask)           vec_loop_stmt(4, _i, _mm256_maskstore_ps(((float *) (ptr)) + 8*_i, (mask).v[_i], (vec).v[_i]);)
#define vec_storeu_masked_f32_16(ptr, vec, mask)           vec_loop_stmt(2, _i, _mm256_maskstore_ps(((float *) (ptr)) + 8*_i, (mask).v[_i], (vec).v[_i]);)
#define vec_storeu_masked_f32_8(ptr, vec, mask)            _mm256_maskstore_ps((float *) (ptr), (mask).v, (vec).v)  /* Reversed order of vec and mask vs maskload, just to escape boredom I guess. */
#define vec_storeu_masked_f32_4(ptr, vec, mask)            _mm_maskstore_ps((float *) (ptr), (mask).v, (vec).v)
#define vec_storeu_masked_f32_1(ptr, vec, mask)            do { if ((mask).v) (*((float *) (ptr))) = ((vec).v); } while (0)

#define vec_gather_f32_i32_32(ptr, idx)                    vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_gather_f32_i32_16(ptr, idx)                    vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_gather_f32_i32_8(ptr, idx)                     vec_f32_8( _mm256_i32gather_ps((float *) (ptr), (idx).v, 4) )
#define vec_gather_f32_i32_4(ptr, idx)                     vec_f32_4( _mm_i32gather_ps((float *) (ptr), (idx).v, 4) )
#define vec_gather_f32_i32_1(ptr, idx)                     vec_f32_1( ((float *) (ptr))[(idx).v] )

#define vec_gather_f32_i64_32(ptr, idx)                    vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_gather_f32_i64_16(ptr, idx)                    vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_ps(((float *) (ptr)) + 8*_i);)
#define vec_gather_f32_i64_8(ptr, idx)                     vec_f32_8( _mm256_i64gather_ps((float *) (ptr), (idx).v, 4) )
#define vec_gather_f32_i64_4(ptr, idx)                     vec_f32_4( _mm_i64gather_ps((float *) (ptr), (idx).v, 4) )
#define vec_gather_f32_i64_1(ptr, idx)                     vec_f32_1( ((float *) (ptr))[(idx).v] )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_add_ps((a).v[_i], (b).v[_i]);)
#define vec_add_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_add_ps((a).v[_i], (b).v[_i]);)
#define vec_add_f32_8(a, b)                                vec_f32_8( _mm256_add_ps((a).v, (b).v) )
#define vec_add_f32_4(a, b)                                vec_f32_4( _mm_add_ps((a).v, (b).v) )
#define vec_add_f32_1(a, b)                                vec_f32_1( ((a).v + (b).v) )

#define vec_sub_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_sub_ps((a).v[_i], (b).v[_i]);)
#define vec_sub_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_sub_ps((a).v[_i], (b).v[_i]);)
#define vec_sub_f32_8(a, b)                                vec_f32_8( _mm256_sub_ps((a).v, (b).v) )
#define vec_sub_f32_4(a, b)                                vec_f32_4( _mm_sub_ps((a).v, (b).v) )
#define vec_sub_f32_1(a, b)                                vec_f32_1( ((a).v - (b).v) )

#define vec_mul_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_mul_ps((a).v[_i], (b).v[_i]);)
#define vec_mul_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_mul_ps((a).v[_i], (b).v[_i]);)
#define vec_mul_f32_8(a, b)                                vec_f32_8( _mm256_mul_ps((a).v, (b).v) )
#define vec_mul_f32_4(a, b)                                vec_f32_4( _mm_mul_ps((a).v, (b).v) )
#define vec_mul_f32_1(a, b)                                vec_f32_1( ((a).v * (b).v) )

#define vec_div_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_div_ps((a).v[_i], (b).v[_i]);)
#define vec_div_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_div_ps((a).v[_i], (b).v[_i]);)
#define vec_div_f32_8(a, b)                                vec_f32_8( _mm256_div_ps((a).v, (b).v) )
#define vec_div_f32_4(a, b)                                vec_f32_4( _mm_div_ps((a).v, (b).v) )
#define vec_div_f32_1(a, b)                                vec_f32_1( ((a).v / (b).v) )

#define vec_fmadd_f32_32(a, b, c)                          vec_loop_expr(vec_f32_32_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_fmadd_ps((a).v[_i], (b).v[_i], (c).v[_i]);)
#define vec_fmadd_f32_16(a, b, c)                          vec_loop_expr(vec_f32_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm256_fmadd_ps((a).v[_i], (b).v[_i], (c).v[_i]);)
#define vec_fmadd_f32_8(a, b, c)                           vec_f32_8( _mm256_fmadd_ps((a).v, (b).v, (c).v) )
#define vec_fmadd_f32_4(a, b, c)                           vec_f32_4( _mm_fmadd_ps((a).v, (b).v, (c).v) )
#define vec_fmadd_f32_1(a, b, c)                           vec_f32_1( ((a).v * (b).v + (c).v) )


#define vec_reduce_add_f32_32(a)                           vec_loop_expr_init(float, 4, _tmp, 0, _i, _tmp += vec_reduce_add_f32_8(vec_f32_8((a).v[_i]));)
#define vec_reduce_add_f32_16(a)                           vec_loop_expr_init(float, 2, _tmp, 0, _i, _tmp += vec_reduce_add_f32_8(vec_f32_8((a).v[_i]));)

#define vec_reduce_add_f32_8(a)                                          \
({                                                                       \
	const __m128 low_128  = _mm256_castps256_ps128((a).v);             \
	const __m128 high_128 = _mm256_extractf128_ps((a).v, 1);           \
	const __m128 low_64   = _mm_add_ps(low_128, high_128);           \
	const __m128 high_64  = _mm_movehl_ps(low_64, low_64);           \
	const __m128 low_32   = _mm_add_ps(low_64, high_64);             \
	const __m128 high_32  = _mm_shuffle_ps(low_32, low_32, 0x01);    \
	const __m128 x32      = _mm_add_ss(low_32, high_32);             \
	_mm_cvtss_f32(x32);                                              \
})

#define vec_reduce_add_f32_4(a)                                          \
({                                                                       \
	const __m128 low_64   = (a).v;                                     \
	const __m128 high_64  = _mm_movehl_ps(low_64, low_64);           \
	const __m128 low_32   = _mm_add_ps(low_64, high_64);             \
	const __m128 high_32  = _mm_shuffle_ps(low_32, low_32, 0x01);    \
	const __m128 x32      = _mm_add_ss(low_32, high_32);             \
	_mm_cvtss_f32(x32);                                              \
})

#define vec_reduce_add_f32_1(a)                            ((a).v)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cmpeq_f32_32(a, b)                             vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, (_tmp).vf[_i] = _mm256_cmp_ps((a).v[_i], (b).v[_i], _CMP_EQ_OS);)
#define vec_cmpeq_f32_16(a, b)                             vec_loop_expr(vec_mask_m32_16_t, 2, _tmp, _i, (_tmp).vf[_i] = _mm256_cmp_ps((a).v[_i], (b).v[_i], _CMP_EQ_OS);)
#define vec_cmpeq_f32_8(a, b)                              vec_mask_m32_8( vec_f32_8(_mm256_cmp_ps((a).v, (b).v, _CMP_EQ_OS)).vi )
#define vec_cmpeq_f32_4(a, b)                              vec_mask_m32_4( vec_f32_4(_mm_cmp_ps((a).v, (b).v, _CMP_EQ_OS)).vi )
#define vec_cmpeq_f32_1(a, b)                              vec_mask_m32_1( (a).v == (b).v )

#define vec_cmpgt_f32_32(a, b)                             vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, (_tmp).vf[_i] = _mm256_cmp_ps((a).v[_i], (b).v[_i], _CMP_GT_OS);)
#define vec_cmpgt_f32_16(a, b)                             vec_loop_expr(vec_mask_m32_16_t, 2, _tmp, _i, (_tmp).vf[_i] = _mm256_cmp_ps((a).v[_i], (b).v[_i], _CMP_GT_OS);)
#define vec_cmpgt_f32_8(a, b)                              vec_mask_m32_8( vec_f32_8(_mm256_cmp_ps((a).v, (b).v, _CMP_GT_OS)).vi )
#define vec_cmpgt_f32_4(a, b)                              vec_mask_m32_4( vec_f32_4(_mm_cmp_ps((a).v, (b).v, _CMP_GT_OS)).vi )
#define vec_cmpgt_f32_1(a, b)                              vec_mask_m32_1( (a).v > (b).v )


#endif /* VECTORIZATION_X86_AVX256_F32_H */

