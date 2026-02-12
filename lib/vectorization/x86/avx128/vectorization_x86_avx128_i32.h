#ifndef VECTORIZATION_X86_AVX128_I32_H
#define VECTORIZATION_X86_AVX128_I32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef union __attribute__((packed, aligned(4))) { int32_t v;    int32_t s[1];  int32_t si[1];  uint32_t su[1];  float sf[1];  }  vec_i32_1_t;
typedef union __attribute__((packed, aligned(4))) { __m128i v;    int32_t s[4];  int32_t si[4];  uint32_t su[4];  float sf[4];  }  vec_i32_4_t;
typedef union __attribute__((packed, aligned(4))) { __m128i v[2]; int32_t s[8];  int32_t si[8];  uint32_t su[8];  float sf[8];  }  vec_i32_8_t;
typedef union __attribute__((packed, aligned(4))) { __m128i v[4]; int32_t s[16]; int32_t si[16]; uint32_t su[16]; float sf[16]; }  vec_i32_16_t;

#define vec_len_default_i32  4


#define vec_i32_4(val)                                     ( (vec_i32_4_t)  { .v = (typeof((vec_i32_4_t){0}.v)) val } )
#define vec_i32_1(val)                                     ( (vec_i32_1_t)  { .v = (typeof((vec_i32_1_t){0}.v)) val } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_array_i32_16(val)                              val.s
#define vec_array_i32_8(val)                               val.s
#define vec_array_i32_4(val)                               val.s
#define vec_array_i32_1(val)                               val.s

#define vec_set1_i32_16(val)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_set1_i32_4(val);)
#define vec_set1_i32_8(val)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_set1_i32_4(val);)
#define vec_set1_i32_4(val)                                vec_i32_4( _mm_set1_epi32(val) )
#define vec_set1_i32_1(val)                                vec_i32_1( val )

#define vec_set_iter_i32_16(iter, expr)                    vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i].v = _mm_set_epi32(vec_iter_expr_4(iter, 4*_i, expr));)
#define vec_set_iter_i32_8(iter, expr)                     vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i].v = _mm_set_epi32(vec_iter_expr_4(iter, 4*_i, expr));)
#define vec_set_iter_i32_4(iter, expr)                     vec_i32_4( _mm_set_epi32(vec_iter_expr_4(iter, 0, expr)) )
#define vec_set_iter_i32_1(iter, expr)                     vec_i32_1( vec_iter_expr_1(iter, 0, expr) )

#define vec_loadu_i32_16(ptr)                              vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_loadu_i32_4(((__m128i *)(ptr)) + _i);)
#define vec_loadu_i32_8(ptr)                               vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_loadu_i32_4(((__m128i *)(ptr)) + _i);)
#define vec_loadu_i32_4(ptr)                               vec_i32_4( _mm_loadu_si128((__m128i *) ptr) )
#define vec_loadu_i32_1(ptr)                               vec_i32_1( (*((uint32_t *) ptr)) )

#define vec_storeu_i32_16(ptr, vec)                        vec_loop_stmt(4, _i, vec_storeu_i32_4(((__m128i *)(ptr)) + _i, vec.v[_i]);)
#define vec_storeu_i32_8(ptr, vec)                         vec_loop_stmt(2, _i, vec_storeu_i32_4(((__m128i *)(ptr)) + _i, vec.v[_i]);)
#define vec_storeu_i32_4(ptr, vec)                         _mm_storeu_si128((__m128i *) ptr, vec.v)
#define vec_storeu_i32_1(ptr, vec)                         do { (*((uint32_t *) ptr)) = (vec.v); } while (0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_add_i32_4(a.v[_i], b.v[_i]);)
#define vec_add_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_add_i32_4(a.v[_i], b.v[_i]);)
#define vec_add_i32_4(a, b)                                vec_i32_4( _mm_add_epi32(a.v, b.v) )
#define vec_add_i32_1(a, b)                                vec_i32_1( (a.v + b.v) )

#define vec_sub_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_sub_i32_4(a.v[_i], b.v[_i]);)
#define vec_sub_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_sub_i32_4(a.v[_i], b.v[_i]);)
#define vec_sub_i32_4(a, b)                                vec_i32_4( _mm_sub_epi32(a.v, b.v) )
#define vec_sub_i32_1(a, b)                                vec_i32_1( (a.v - b.v) )

#define vec_mul_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_mul_i32_4(a.v[_i], b.v[_i]);)
#define vec_mul_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_mul_i32_4(a.v[_i], b.v[_i]);)
#define vec_mul_i32_4(a, b)                                vec_set_iter_i32_4(_j, a.si[_j]*b.si[_j])
#define vec_mul_i32_1(a, b)                                vec_i32_1( (a.v * b.v) )

#define vec_div_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_div_i32_4(a.v[_i], b.v[_i]);)
#define vec_div_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_div_i32_4(a.v[_i], b.v[_i]);)
#define vec_div_i32_4(a, b)                                vec_set_iter_i32_4(_j, a.si[_j]/b.si[_j])
#define vec_div_i32_1(a, b)                                vec_i32_1( (a.v / b.v) )

#define vec_and_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_and_i32_4(a.v[_i], b.v[_i]);)
#define vec_and_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_and_i32_4(a.v[_i], b.v[_i]);)
#define vec_and_i32_4(a, b)                                vec_i32_4( _mm_and_si128(a.v, b.v) )
#define vec_and_i32_1(a, b)                                vec_i32_1( (a.v & b.v) )

#define vec_or_i32_16(a, b)                                vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_or_i32_4(a.v[_i], b.v[_i]);)
#define vec_or_i32_8(a, b)                                 vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_or_i32_4(a.v[_i], b.v[_i]);)
#define vec_or_i32_4(a, b)                                 vec_i32_4( _mm_or_si128(a.v, b.v) )
#define vec_or_i32_1(a, b)                                 vec_i32_1( (a.v | b.v) )

#define vec_xor_i32_16(a, b)                               vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_xor_i32_4(a.v[_i], b.v[_i]);)
#define vec_xor_i32_8(a, b)                                vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_xor_i32_4(a.v[_i], b.v[_i]);)
#define vec_xor_i32_4(a, b)                                vec_i32_4( _mm_xor_si128(a.v, b.v) )
#define vec_xor_i32_1(a, b)                                vec_i32_1( (a.v ^ b.v) )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Shift
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_slli_i32_16(a, imm8)                           vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_slli_i32_4(a.v[_i], imm8);)
#define vec_slli_i32_8(a, imm8)                            vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_slli_i32_4(a.v[_i], imm8);)
#define vec_slli_i32_4(a, imm8)                            vec_i32_4( _mm_slli_epi32(a.v, imm8) )
#define vec_slli_i32_1(a, imm8)                            vec_i32_1( ((imm8 < 32) ? ((uint32_t) a.v)<<imm8 : 0) )

#define vec_srli_i32_16(a, imm8)                           vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_srli_i32_4(a.v[_i], imm8);)
#define vec_srli_i32_8(a, imm8)                            vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_srli_i32_4(a.v[_i], imm8);)
#define vec_srli_i32_4(a, imm8)                            vec_i32_4( _mm_srli_epi32(a.v, imm8) )
#define vec_srli_i32_1(a, imm8)                            vec_i32_1( ((imm8 < 32) ? ((uint32_t) a.v)>>imm8 : 0) )

#define vec_srai_i32_16(a, imm8)                           vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_srai_i32_4(a.v[_i], imm8);)
#define vec_srai_i32_8(a, imm8)                            vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_srai_i32_4(a.v[_i], imm8);)
#define vec_srai_i32_4(a, imm8)                            vec_set_iter_i32_4(_j, vec_srai_i32_s(a.si[_j], imm8))
#define vec_srai_i32_1(a, imm8)                            vec_set_iter_i32_1(_j, vec_srai_i32_s(a.si[_j], imm8))
#define vec_srai_i32_s(a, imm8)                            ((imm8 < 32) ? ((int32_t) a)>>imm8 : (((int32_t) a) < 0) ? -1 : 0)

#define vec_sllv_i32_16(a, count)                          vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_sllv_i32_4(a.v[_i], count.v[_i]);)
#define vec_sllv_i32_8(a, count)                           vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_sllv_i32_4(a.v[_i], count.v[_i]);)
#define vec_sllv_i32_4(a, count)                           vec_set_iter_i32_4(_j, vec_sllv_i32_s(a.si[_j], count.si[_j]))
#define vec_sllv_i32_1(a, count)                           vec_set_iter_i32_1(_j, vec_sllv_i32_s(a.si[_j], count.si[_j]))
#define vec_sllv_i32_s(a, count)                           ((count < 32) ? ((uint32_t) a)<<count : 0)

#define vec_srlv_i32_16(a, count)                          vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_srlv_i32_4(a.v[_i], count.v[_i]);)
#define vec_srlv_i32_8(a, count)                           vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_srlv_i32_4(a.v[_i], count.v[_i]);)
#define vec_srlv_i32_4(a, count)                           vec_set_iter_i32_4(_j, vec_srlv_i32_s(a.si[_j], count.si[_j]))
#define vec_srlv_i32_1(a, count)                           vec_set_iter_i32_1(_j, vec_srlv_i32_s(a.si[_j], count.si[_j]))
#define vec_srlv_i32_s(a, count)                           ((count < 32) ? ((uint32_t) a)>>count : 0)

#define vec_srav_i32_16(a, count)                          vec_loop_expr(vec_i32_16_t, 4, _tmp, _i, _tmp.v[_i] = vec_srav_i32_4(a.v[_i], count.v[_i]);)
#define vec_srav_i32_8(a, count)                           vec_loop_expr(vec_i32_8_t, 2, _tmp, _i, _tmp.v[_i] = vec_srav_i32_4(a.v[_i], count.v[_i]);)
#define vec_srav_i32_4(a, count)                           vec_set_iter_i32_4(_j, vec_srav_i32_s(a.si[_j], count.si[_j]))
#define vec_srav_i32_1(a, count)                           vec_set_iter_i32_1(_j, vec_srav_i32_s(a.si[_j], count.si[_j]))
#define vec_srav_i32_s(a, count)                           ((count < 32) ? ((int32_t) a)>>count : (((int32_t) a) < 0) ? -1 : 0)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------------------
//- Shuffle - Permute
//------------------------------------------------------------------------------------------------------------------------------------------

#endif /* VECTORIZATION_X86_AVX128_I32_H */

