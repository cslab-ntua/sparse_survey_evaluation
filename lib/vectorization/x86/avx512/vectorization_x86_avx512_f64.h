#ifndef VECTORIZATION_X86_AVX512_F64_H
#define VECTORIZATION_X86_AVX512_F64_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"
#include "vectorization_x86_avx512_m64.h"


typedef union __attribute__((packed, aligned(8))) { double  v;    float  vf32;    int64_t vi;     double s[ 1]; int64_t si[ 1]; uint64_t su[ 1]; double sf[ 1]; }  vec_f64_1_t;
typedef union __attribute__((packed, aligned(8))) { __m128d v;    __m128 vf32;    __m128i vi;     double s[ 2]; int64_t si[ 2]; uint64_t su[ 2]; double sf[ 2]; }  vec_f64_2_t;
typedef union __attribute__((packed, aligned(8))) { __m256d v;    __m256 vf32;    __m256i vi;     double s[ 4]; int64_t si[ 4]; uint64_t su[ 4]; double sf[ 4]; }  vec_f64_4_t;
typedef union __attribute__((packed, aligned(8))) { __m512d v;    __m512 vf32;    __m512i vi;     double s[ 8]; int64_t si[ 8]; uint64_t su[ 8]; double sf[ 8]; }  vec_f64_8_t;
typedef union __attribute__((packed, aligned(8))) { __m512d v[2]; __m512 vf32[2]; __m512i vi[2];  double s[16]; int64_t si[16]; uint64_t su[16]; double sf[16]; }  vec_f64_16_t;

#define vec_len_default_f64     8
#define vec_len_default_f64_f2  4


#define vec_f64_8(val)                                     ( (vec_f64_8_t)  { .v = (val) } )
#define vec_f64_4(val)                                     ( (vec_f64_4_t)  { .v = (val) } )
#define vec_f64_2(val)                                     ( (vec_f64_2_t)  { .v = (val) } )
#define vec_f64_1(val)                                     ( (vec_f64_1_t)  { .v = (val) } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Cast
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cast_to_i32_f64_16(val)                        vec_loop_expr(vec_i32_32_t, 2, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i32_f64_8(val)                         ( (vec_i32_16_t) { .v = (val).vi } )
#define vec_cast_to_i32_f64_4(val)                         ( (vec_i32_8_t)  { .v = (val).vi } )
#define vec_cast_to_i32_f64_2(val)                         ( (vec_i32_4_t)  { .v = (val).vi } )
#define vec_cast_to_i32_f64_1(val)                         ( (vec_i32_2_t)  { .v = (val).vi } )

#define vec_cast_to_i64_f64_16(val)                        vec_loop_expr(vec_i64_16_t, 2, _tmp, _i, (_tmp).v[_i] = (val).vi[_i];)
#define vec_cast_to_i64_f64_8(val)                         ( (vec_i64_8_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_4(val)                         ( (vec_i64_4_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_2(val)                         ( (vec_i64_2_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_1(val)                         ( (vec_i64_1_t)  { .v = (val).vi } )

#define vec_cast_to_f32_f64_16(val)                        vec_loop_expr(vec_f32_32_t, 2, _tmp, _i, (_tmp).v[_i] = (val).vf32[_i];)
#define vec_cast_to_f32_f64_8(val)                         ( (vec_f32_16_t) { .v = (val).vf32 } )
#define vec_cast_to_f32_f64_4(val)                         ( (vec_f32_8_t)  { .v = (val).vf32 } )
#define vec_cast_to_f32_f64_2(val)                         ( (vec_f32_4_t)  { .v = (val).vf32 } )
#define vec_cast_to_f32_f64_1(val)                         ( (vec_f32_2_t)  { .v = (val).vf32 } )

#define vec_cast_to_f64_f64_16(val)                        (val)
#define vec_cast_to_f64_f64_8(val)                         (val)
#define vec_cast_to_f64_f64_4(val)                         (val)
#define vec_cast_to_f64_f64_2(val)                         (val)
#define vec_cast_to_f64_f64_1(val)                         (val)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_elem_get_f64_16(vec, index)                    ((vec).s[index])
#define vec_elem_get_f64_8(vec, index)                     ((vec).s[index])
#define vec_elem_get_f64_4(vec, index)                     ((vec).s[index])
#define vec_elem_get_f64_2(vec, index)                     ((vec).s[index])
#define vec_elem_get_f64_1(vec, index)                     ((vec).s[index])

#define vec_elem_set_f64_16(vec, index, expr)              { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f64_8(vec, index, expr)               { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f64_4(vec, index, expr)               { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f64_2(vec, index, expr)               { (vec).s[index] = (expr); } while (0)
#define vec_elem_set_f64_1(vec, index, expr)               { (vec).s[index] = (expr); } while (0)

#define vec_array_f64_16(vec)                              (vec).s
#define vec_array_f64_8(vec)                               (vec).s
#define vec_array_f64_4(vec)                               (vec).s
#define vec_array_f64_2(vec)                               (vec).s
#define vec_array_f64_1(vec)                               (vec).s

#define vec_set1_f64_16(val)                               vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_set1_pd(val);)
#define vec_set1_f64_8(val)                                vec_f64_8( _mm512_set1_pd(val) )
#define vec_set1_f64_4(val)                                vec_f64_4( _mm256_set1_pd(val) )
#define vec_set1_f64_2(val)                                vec_f64_2( _mm_set1_pd(val) )
#define vec_set1_f64_1(val)                                vec_f64_1( (double) (val) )

#define vec_set_iter_f64_16(iter, expr)                    vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_set_pd(vec_iter_expr_8(iter, 8*_i, (expr)));)
#define vec_set_iter_f64_8(iter, expr)                     vec_f64_8( _mm512_set_pd(vec_iter_expr_8(iter, 0, (expr))) )
#define vec_set_iter_f64_4(iter, expr)                     vec_f64_4( _mm256_set_pd(vec_iter_expr_4(iter, 0, (expr))) )
#define vec_set_iter_f64_2(iter, expr)                     vec_f64_2( _mm_set_pd(vec_iter_expr_2(iter, 0, (expr))) )
#define vec_set_iter_f64_1(iter, expr)                     vec_f64_1( (double) vec_iter_expr_1(iter, 0, (expr)) )

#define vec_loadu_f64_16(ptr)                              vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_loadu_pd(((double *) (ptr)) + 8*_i);)
#define vec_loadu_f64_8(ptr)                               vec_f64_8( _mm512_loadu_pd((double *) (ptr)) )
#define vec_loadu_f64_4(ptr)                               vec_f64_4( _mm256_loadu_pd((double *) (ptr)) )
#define vec_loadu_f64_2(ptr)                               vec_f64_2( _mm_loadu_pd((double *) (ptr)) )
#define vec_loadu_f64_1(ptr)                               vec_f64_1( (*((double *) (ptr))) )

#define vec_loadu_maskedz_f64_16(ptr, mask)                vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_maskz_loadu_pd((mask).v[_i], ((__m512d *) (ptr)) + _i);)
#define vec_loadu_maskedz_f64_8(ptr, mask)                 vec_f64_8( _mm512_maskz_loadu_pd((mask).v, (__m512d *) (ptr)) )
#define vec_loadu_maskedz_f64_4(ptr, mask)                 vec_f64_4( _mm256_maskz_loadu_pd((mask).v, (__m256d *) (ptr)) )
#define vec_loadu_maskedz_f64_2(ptr, mask)                 vec_f64_2( _mm_maskz_loadu_pd((mask).v, (__m128d *) (ptr)) )
#define vec_loadu_maskedz_f64_1(ptr, mask)                 vec_f64_1( ((mask).v) ? (*((double *) (ptr))) : 0 )

#define vec_storeu_f64_16(ptr, vec)                        vec_loop_stmt(2, _i, _mm512_storeu_pd(((double *) (ptr)) + 8*_i, (vec).v[_i]);)
#define vec_storeu_f64_8(ptr, vec)                         _mm512_storeu_pd((double *) (ptr), (vec).v)
#define vec_storeu_f64_4(ptr, vec)                         _mm256_storeu_pd((double *) (ptr), (vec).v)
#define vec_storeu_f64_2(ptr, vec)                         _mm_storeu_pd((double *) (ptr), (vec).v)
#define vec_storeu_f64_1(ptr, vec)                         do { (*((double *) (ptr))) = ((vec).v); } while (0)

#define vec_storeu_masked_f64_16(ptr, vec, mask)           vec_loop_stmt(2, _i, _mm512_mask_storeu_pd(((__m512d *) (ptr)) + _i, (mask).v[_i], (vec).v[_i]);)
#define vec_storeu_masked_f64_8(ptr, vec, mask)            _mm512_mask_storeu_pd((__m512d *) (ptr), (mask).v, (vec).v)
#define vec_storeu_masked_f64_4(ptr, vec, mask)            _mm256_mask_storeu_pd((__m256d *) (ptr), (mask).v, (vec).v)
#define vec_storeu_masked_f64_2(ptr, vec, mask)            _mm_mask_storeu_pd((__m128d *) (ptr), (mask).v, (vec).v)
#define vec_storeu_masked_f64_1(ptr, vec, mask)            do { if ((mask).v) (*((double *) (ptr))) = ((vec).v); } while (0)

#define vec_gather_f64_i32_16(ptr, idx)                    vec_loop_expr(vec_f64_16_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_pd(((double *) (ptr)) + 4*_i);)
#define vec_gather_f64_i32_8(ptr, idx)                     vec_f64_8( _mm512_i32gather_pd((idx).v, (double *) (ptr), 8) )
#define vec_gather_f64_i32_4(ptr, idx)                     vec_f64_4( _mm256_i32gather_pd((double *) (ptr), (idx).v, 8) )
#define vec_gather_f64_i32_2(ptr, idx)                     vec_f64_2( _mm_i32gather_pd((double *) (ptr), (idx).v, 8) )
#define vec_gather_f64_i32_1(ptr, idx)                     vec_f64_1( ((double *) (ptr))[(idx).v] )

#define vec_gather_f64_i64_16(ptr, idx)                    vec_loop_expr(vec_f64_16_t, 4, _tmp, _i, (_tmp).v[_i] = _mm256_loadu_pd(((double *) (ptr)) + 4*_i);)
#define vec_gather_f64_i64_8(ptr, idx)                     vec_f64_8( _mm512_i64gather_pd((idx).v, (double *) (ptr), 8) )
#define vec_gather_f64_i64_4(ptr, idx)                     vec_f64_4( _mm256_i64gather_pd((double *) (ptr), (idx).v, 8) )
#define vec_gather_f64_i64_2(ptr, idx)                     vec_f64_2( _mm_i64gather_pd((double *) (ptr), (idx).v, 8) )
#define vec_gather_f64_i64_1(ptr, idx)                     vec_f64_1( ((double *) (ptr))[(idx).v] )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_add_pd((a).v[_i], (b).v[_i]);)
#define vec_add_f64_8(a, b)                                vec_f64_8( _mm512_add_pd((a).v, (b).v) )
#define vec_add_f64_4(a, b)                                vec_f64_4( _mm256_add_pd((a).v, (b).v) )
#define vec_add_f64_2(a, b)                                vec_f64_2( _mm_add_pd((a).v, (b).v) )
#define vec_add_f64_1(a, b)                                vec_f64_1( (a).v + (b).v )

#define vec_sub_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_sub_pd((a).v[_i], (b).v[_i]);)
#define vec_sub_f64_8(a, b)                                vec_f64_8( _mm512_sub_pd((a).v, (b).v) )
#define vec_sub_f64_4(a, b)                                vec_f64_4( _mm256_sub_pd((a).v, (b).v) )
#define vec_sub_f64_2(a, b)                                vec_f64_2( _mm_sub_pd((a).v, (b).v) )
#define vec_sub_f64_1(a, b)                                vec_f64_1( (a).v - (b).v )

#define vec_mul_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_mul_pd((a).v[_i], (b).v[_i]);)
#define vec_mul_f64_8(a, b)                                vec_f64_8( _mm512_mul_pd((a).v, (b).v) )
#define vec_mul_f64_4(a, b)                                vec_f64_4( _mm256_mul_pd((a).v, (b).v) )
#define vec_mul_f64_2(a, b)                                vec_f64_2( _mm_mul_pd((a).v, (b).v) )
#define vec_mul_f64_1(a, b)                                vec_f64_1( (a).v * (b).v )

#define vec_div_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_div_pd((a).v[_i], (b).v[_i]);)
#define vec_div_f64_8(a, b)                                vec_f64_8( _mm512_div_pd((a).v, (b).v) )
#define vec_div_f64_4(a, b)                                vec_f64_4( _mm256_div_pd((a).v, (b).v) )
#define vec_div_f64_2(a, b)                                vec_f64_2( _mm_div_pd((a).v, (b).v) )
#define vec_div_f64_1(a, b)                                vec_f64_1( (a).v / (b).v )

#define vec_fmadd_f64_16(a, b, c)                          vec_loop_expr(vec_f64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_fmadd_pd((a).v[_i], (b).v[_i], (c).v[_i]);)
#define vec_fmadd_f64_8(a, b, c)                           vec_f64_8( _mm512_fmadd_pd((a).v, (b).v, (c).v) )
#define vec_fmadd_f64_4(a, b, c)                           vec_f64_4( _mm256_fmadd_pd((a).v, (b).v, (c).v) )
#define vec_fmadd_f64_2(a, b, c)                           vec_f64_2( _mm_fmadd_pd((a).v, (b).v, (c).v) )
#define vec_fmadd_f64_1(a, b, c)                           vec_f64_1( (a).v * (b).v + (c).v )


#define vec_reduce_add_f64_16(a)                           vec_loop_expr_init(double, 2, _tmp, 0, _i, _tmp += vec_reduce_add_f64_8(vec_f64_8((a).v[_i]));)

#define vec_reduce_add_f64_8(a)                            _mm512_reduce_add_pd((a).v)

#define vec_reduce_add_f64_4(a)                                                                                                                                                                                                       \
({                                                                                                                                                                                                                                    \
	__m128d low_128d  = _mm256_castpd256_pd128((a).v);      /* Cast vector of type __m256d to type __m128d. This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency. */      \
	__m128d high_128d = _mm256_extractf128_pd((a).v, 1);    /* High 128: Extract 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a, selected with imm8, and store the result in dst. */    \
	low_128d  = _mm_add_pd(low_128d, high_128d);          /* Add low 128 and high 128. */                                                                                                                                         \
	__m128d high64 = _mm_unpackhi_pd(low_128d, low_128d); /* High 64: Unpack and interleave double-precision (64-bit) floating-point elements from the high half of a and b, and store the results in dst. */                     \
	_mm_cvtsd_f64(_mm_add_sd(low_128d, high64));          /* Reduce to scalar. */                                                                                                                                                 \
})

#define vec_reduce_add_f64_2(a)                            \
({                                                         \
	__m128d high64 = _mm_unpackhi_pd((a).v, (a).v);    \
	_mm_cvtsd_f64(_mm_add_sd((a).v, high64));          \
})

#define vec_reduce_add_f64_1(a)                            ((a).v)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cmpeq_f64_16(a, b)                             vec_loop_expr(vec_mask_m64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_cmp_pd_mask((a).v[_i], (b).v[_i], _CMP_EQ_OS);)
#define vec_cmpeq_f64_8(a, b)                              vec_mask_m64_8( _mm512_cmp_pd_mask((a).v, (b).v, _CMP_EQ_OS) )
#define vec_cmpeq_f64_4(a, b)                              vec_mask_m64_4( _mm256_cmp_pd_mask((a).v, (b).v, _CMP_EQ_OS) )
#define vec_cmpeq_f64_2(a, b)                              vec_mask_m64_2( _mm_cmp_pd_mask((a).v, (b).v, _CMP_EQ_OS) )
#define vec_cmpeq_f64_1(a, b)                              vec_mask_m64_1( (__mmask8) ((a).v == (b).v) )

#define vec_cmpgt_f64_16(a, b)                             vec_loop_expr(vec_mask_m64_16_t, 2, _tmp, _i, (_tmp).v[_i] = _mm512_cmp_pd_mask((a).v[_i], (b).v[_i], _CMP_GT_OS);)
#define vec_cmpgt_f64_8(a, b)                              vec_mask_m64_8( _mm512_cmp_pd_mask((a).v, (b).v, _CMP_GT_OS) )
#define vec_cmpgt_f64_4(a, b)                              vec_mask_m64_4( _mm256_cmp_pd_mask((a).v, (b).v, _CMP_GT_OS) )
#define vec_cmpgt_f64_2(a, b)                              vec_mask_m64_2( _mm_cmp_pd_mask((a).v, (b).v, _CMP_GT_OS) )
#define vec_cmpgt_f64_1(a, b)                              vec_mask_m64_1( (__mmask8) ((a).v > (b).v) )


#endif /* VECTORIZATION_X86_AVX512_F64_H */

