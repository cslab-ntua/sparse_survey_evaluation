#ifndef VECTORIZATION_X86_AVX128_I64_H
#define VECTORIZATION_X86_AVX128_I64_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef union __attribute__((packed, aligned(8))) { int64_t v;    int64_t s[1]; int64_t si[1]; uint64_t su[1]; double sf[1]; }  vec_i64_1_t;
typedef union __attribute__((packed, aligned(8))) { __m128i v;    int64_t s[2]; int64_t si[2]; uint64_t su[2]; double sf[2]; }  vec_i64_2_t;
typedef union __attribute__((packed, aligned(8))) { __m128i v[2]; int64_t s[4]; int64_t si[4]; uint64_t su[4]; double sf[4]; }  vec_i64_4_t;
typedef union __attribute__((packed, aligned(8))) { __m128i v[4]; int64_t s[8]; int64_t si[8]; uint64_t su[8]; double sf[8]; }  vec_i64_8_t;

#define vec_len_default_i64  2


#define vec_i64_2(val)                                     ( (vec_i64_2_t)  { .v = (typeof((vec_i64_2_t){0}.v)) val } )
#define vec_i64_1(val)                                     ( (vec_i64_1_t)  { .v = (typeof((vec_i64_1_t){0}.v)) val } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_array_i64_8(val)                               val.s
#define vec_array_i64_4(val)                               val.s
#define vec_array_i64_2(val)                               val.s
#define vec_array_i64_1(val)                               val.s

#define vec_set1_i64_8(val)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_set1_i64_2(val);)
#define vec_set1_i64_4(val)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_set1_i64_2(val);)
#define vec_set1_i64_2(val)                                vec_i64_2( _mm_set1_epi64x(val) )
#define vec_set1_i64_1(val)                                vec_i64_1( val )

#define vec_set_iter_i64_8(iter, expr)                     vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i].v = _mm_set_epi64x(vec_iter_expr_2(iter, 2*_i, expr));)
#define vec_set_iter_i64_4(iter, expr)                     vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i].v = _mm_set_epi64x(vec_iter_expr_2(iter, 2*_i, expr));)
#define vec_set_iter_i64_2(iter, expr)                     vec_i64_2( _mm_set_epi64x(vec_iter_expr_2(iter, 0, expr)) )
#define vec_set_iter_i64_1(iter, expr)                     vec_i64_1( vec_iter_expr_1(iter, 0, expr) )

#define vec_loadu_i64_8(ptr)                               vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_loadu_i64_2(((__m128i *)(ptr)) + _i);)
#define vec_loadu_i64_4(ptr)                               vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_loadu_i64_2(((__m128i *)(ptr)) + _i);)
#define vec_loadu_i64_2(ptr)                               vec_i64_2( _mm_loadu_si128((__m128i *) ptr) )
#define vec_loadu_i64_1(ptr)                               vec_i64_1( (*((uint64_t *) ptr)) )

#define vec_storeu_i64_8(ptr, vec)                         vec_loop_stmt(4, _i, vec_storeu_i64_2(((__m128i *)(ptr)) + _i, vec.v[_i]);)
#define vec_storeu_i64_4(ptr, vec)                         vec_loop_stmt(2, _i, vec_storeu_i64_2(((__m128i *)(ptr)) + _i, vec.v[_i]);)
#define vec_storeu_i64_2(ptr, vec)                         _mm_storeu_si128((__m128i *) ptr, vec.v)
#define vec_storeu_i64_1(ptr, vec)                         do { (*((uint64_t *) ptr)) = (vec.v); } while (0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_add_i64_2(a.v[_i], b.v[_i]);)
#define vec_add_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_add_i64_2(a.v[_i], b.v[_i]);)
#define vec_add_i64_2(a, b)                                vec_i64_2( _mm_add_epi64(a.v, b.v) )
#define vec_add_i64_1(a, b)                                vec_i64_1( (a.v + b.v) )

#define vec_sub_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_sub_i64_2(a.v[_i], b.v[_i]);)
#define vec_sub_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_sub_i64_2(a.v[_i], b.v[_i]);)
#define vec_sub_i64_2(a, b)                                vec_i64_2( _mm_sub_epi64(a.v, b.v) )
#define vec_sub_i64_1(a, b)                                vec_i64_1( (a.v - b.v) )

#define vec_mul_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_mul_i64_2(a.v[_i], b.v[_i]);)
#define vec_mul_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_mul_i64_2(a.v[_i], b.v[_i]);)
#define vec_mul_i64_2(a, b)                                vec_set_iter_i64_2(_j, a.si[_j]*b.si[_j])
#define vec_mul_i64_1(a, b)                                vec_i64_1( (a.v * b.v) )

#define vec_div_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_div_i64_2(a.v[_i], b.v[_i]);)
#define vec_div_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_div_i64_2(a.v[_i], b.v[_i]);)
#define vec_div_i64_2(a, b)                                vec_set_iter_i64_2(_j, a.si[_j]/b.si[_j])
#define vec_div_i64_1(a, b)                                vec_i64_1( (a.v / b.v) )

#define vec_and_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_and_i64_2(a.v[_i], b.v[_i]);)
#define vec_and_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_and_i64_2(a.v[_i], b.v[_i]);)
#define vec_and_i64_2(a, b)                                vec_i64_2( _mm_and_si128(a.v, b.v) )
#define vec_and_i64_1(a, b)                                vec_i64_1( (a.v & b.v) )

#define vec_or_i64_8(a, b)                                 vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_or_i64_2(a.v[_i], b.v[_i]);)
#define vec_or_i64_4(a, b)                                 vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_or_i64_2(a.v[_i], b.v[_i]);)
#define vec_or_i64_2(a, b)                                 vec_i64_2( _mm_or_si128(a.v, b.v) )
#define vec_or_i64_1(a, b)                                 vec_i64_1( (a.v | b.v) )

#define vec_xor_i64_8(a, b)                                vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_xor_i64_2(a.v[_i], b.v[_i]);)
#define vec_xor_i64_4(a, b)                                vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_xor_i64_2(a.v[_i], b.v[_i]);)
#define vec_xor_i64_2(a, b)                                vec_i64_2( _mm_xor_si128(a.v, b.v) )
#define vec_xor_i64_1(a, b)                                vec_i64_1( (a.v ^ b.v) )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Shift
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_slli_i64_8(a, imm8)                            vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_slli_i64_2(a.v[_i], imm8);)
#define vec_slli_i64_4(a, imm8)                            vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_slli_i64_2(a.v[_i], imm8);)
#define vec_slli_i64_2(a, imm8)                            vec_i64_2( _mm_slli_epi64(a.v, imm8) )
#define vec_slli_i64_1(a, imm8)                            vec_i64_1( ((imm8 < 64) ? ((uint64_t) a.v)<<imm8 : 0) )

#define vec_srli_i64_8(a, imm8)                            vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_srli_i64_2(a.v[_i], imm8);)
#define vec_srli_i64_4(a, imm8)                            vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_srli_i64_2(a.v[_i], imm8);)
#define vec_srli_i64_2(a, imm8)                            vec_i64_2( _mm_srli_epi64(a.v, imm8) )
#define vec_srli_i64_1(a, imm8)                            vec_i64_1( ((imm8 < 64) ? ((uint64_t) a.v)>>imm8 : 0) )

#define vec_srai_i64_8(a, imm8)                            vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_srai_i64_2(a.v[_i], imm8);)
#define vec_srai_i64_4(a, imm8)                            vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_srai_i64_2(a.v[_i], imm8);)
#define vec_srai_i64_2(a, imm8)                            vec_set_iter_i64_2(_j, vec_srai_i64_s(a.si[_j], imm8))
#define vec_srai_i64_1(a, imm8)                            vec_set_iter_i64_1(_j, vec_srai_i64_s(a.si[_j], imm8))
#define vec_srai_i64_s(a, imm8)                            ((imm8 < 64) ? ((int64_t) a)>>imm8 : (((int64_t) a) < 0) ? -1 : 0)

#define vec_sllv_i64_8(a, count)                           vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_sllv_i64_2(a.v[_i], count.v[_i]);)
#define vec_sllv_i64_4(a, count)                           vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_sllv_i64_2(a.v[_i], count.v[_i]);)
#define vec_sllv_i64_2(a, count)                           vec_set_iter_i64_2(_j, vec_sllv_i64_s(a.si[_j], count.si[_j]))
#define vec_sllv_i64_1(a, count)                           vec_set_iter_i64_1(_j, vec_sllv_i64_s(a.si[_j], count.si[_j]))
#define vec_sllv_i64_s(a, count)                           ((count < 64) ? ((uint64_t) a)<<count : 0)

#define vec_srlv_i64_8(a, count)                           vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_srlv_i64_2(a.v[_i], count.v[_i]);)
#define vec_srlv_i64_4(a, count)                           vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_srlv_i64_2(a.v[_i], count.v[_i]);)
#define vec_srlv_i64_2(a, count)                           vec_set_iter_i64_2(_j, vec_srlv_i64_s(a.si[_j], count.si[_j]))
#define vec_srlv_i64_1(a, count)                           vec_set_iter_i64_1(_j, vec_srlv_i64_s(a.si[_j], count.si[_j]))
#define vec_srlv_i64_s(a, count)                           ((count < 64) ? ((uint64_t) a)>>count : 0)

#define vec_srav_i64_8(a, count)                           vec_loop_expr(vec_i64_8_t, 4, _tmp, _i, _tmp.v[_i] = vec_srav_i64_2(a.v[_i], count.v[_i]);)
#define vec_srav_i64_4(a, count)                           vec_loop_expr(vec_i64_4_t, 2, _tmp, _i, _tmp.v[_i] = vec_srav_i64_2(a.v[_i], count.v[_i]);)
#define vec_srav_i64_2(a, count)                           vec_set_iter_i64_2(_j, vec_srav_i64_s(a.si[_j], count.si[_j]))
#define vec_srav_i64_1(a, count)                           vec_set_iter_i64_1(_j, vec_srav_i64_s(a.si[_j], count.si[_j]))
#define vec_srav_i64_s(a, count)                           ((count < 64) ? ((int64_t) a)>>count : (((int64_t) a) < 0) ? -1 : 0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------------------
//- Shuffle - Permute
//------------------------------------------------------------------------------------------------------------------------------------------

#endif /* VECTORIZATION_X86_AVX128_I64_H */

