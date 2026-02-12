#ifndef VECTORIZATION_SCALAR_F32_H
#define VECTORIZATION_SCALAR_F32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"
#include "vectorization_scalar_m32.h"


typedef union __attribute__((packed, aligned(4))) {         float         v;         double         vf64;         int64_t         vi;  float s[ 1]; int32_t si[ 1]; uint32_t su[ 1]; float sf[ 1]; }  vec_f32_1_t;
typedef union __attribute__((packed, aligned(4))) { struct {float a[ 4];} v; struct {double a[ 2];} vf64; struct {int64_t a[ 2];} vi;  float s[ 4]; int32_t si[ 4]; uint32_t su[ 4]; float sf[ 4]; }  vec_f32_4_t;
typedef union __attribute__((packed, aligned(4))) { struct {float a[ 8];} v; struct {double a[ 4];} vf64; struct {int64_t a[ 4];} vi;  float s[ 8]; int32_t si[ 8]; uint32_t su[ 8]; float sf[ 8]; }  vec_f32_8_t;
typedef union __attribute__((packed, aligned(4))) { struct {float a[16];} v; struct {double a[ 8];} vf64; struct {int64_t a[ 8];} vi;  float s[16]; int32_t si[16]; uint32_t su[16]; float sf[16]; }  vec_f32_16_t;
typedef union __attribute__((packed, aligned(4))) { struct {float a[32];} v; struct {double a[16];} vf64; struct {int64_t a[16];} vi;  float s[32]; int32_t si[32]; uint32_t su[32]; float sf[32]; }  vec_f32_32_t;

#define vec_len_default_f32  8


#define vec_f32_32(val)                                    ( (vec_f32_32_t) { .v = (val) } )
#define vec_f32_16(val)                                    ( (vec_f32_16_t) { .v = (val) } )
#define vec_f32_8(val)                                     ( (vec_f32_8_t)  { .v = (val) } )
#define vec_f32_4(val)                                     ( (vec_f32_4_t)  { .v = (val) } )
#define vec_f32_1(val)                                     ( (vec_f32_1_t)  { .v = (val) } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Cast
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cast_to_i32_f32_32(val)                        ( (vec_i32_32_t) { .v = (val).v } )
#define vec_cast_to_i32_f32_16(val)                        ( (vec_i32_16_t) { .v = (val).v } )
#define vec_cast_to_i32_f32_8(val)                         ( (vec_i32_8_t)  { .v = (val).v } )
#define vec_cast_to_i32_f32_4(val)                         ( (vec_i32_4_t)  { .v = (val).v } )
#define vec_cast_to_i32_f32_1(val)                         ( (vec_i32_1_t)  { .v = (val).v } )

#define vec_cast_to_i64_f32_32(val)                        ( (vec_i64_16_t) { .v = (val).vi } )
#define vec_cast_to_i64_f32_16(val)                        ( (vec_i64_8_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f32_8(val)                         ( (vec_i64_4_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f32_4(val)                         ( (vec_i64_2_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f32_1(val)                         ( (vec_i64_1_t)  { .v = (val).vi } )

#define vec_cast_to_f32_f32_32(val)                        (val)
#define vec_cast_to_f32_f32_16(val)                        (val)
#define vec_cast_to_f32_f32_8(val)                         (val)
#define vec_cast_to_f32_f32_4(val)                         (val)
#define vec_cast_to_f32_f32_1(val)                         (val)

#define vec_cast_to_f64_f32_32(val)                        ( (vec_f64_16_t) { .v = (val).vf64 } )
#define vec_cast_to_f64_f32_16(val)                        ( (vec_f64_8_t)  { .v = (val).vf64 } )
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

#define vec_set1_f32_32(val)                               vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f32_16(val)                               vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f32_8(val)                                vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f32_4(val)                                vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f32_1(val)                                vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = (val);)

#define vec_set_iter_f32_32(iter, expr)                    vec_loop_expr(vec_f32_32_t, 32, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f32_16(iter, expr)                    vec_loop_expr(vec_f32_16_t, 16, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f32_8(iter, expr)                     vec_loop_expr(vec_f32_8_t,   8, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f32_4(iter, expr)                     vec_loop_expr(vec_f32_4_t,   4, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f32_1(iter, expr)                     vec_loop_expr(vec_f32_1_t,   1, _tmp, iter, _tmp.s[iter] = (expr);)

#define vec_loadu_f32_32(ptr)                              vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = ((float *) (ptr))[_i];)
#define vec_loadu_f32_16(ptr)                              vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = ((float *) (ptr))[_i];)
#define vec_loadu_f32_8(ptr)                               vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = ((float *) (ptr))[_i];)
#define vec_loadu_f32_4(ptr)                               vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = ((float *) (ptr))[_i];)
#define vec_loadu_f32_1(ptr)                               vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = ((float *) (ptr))[_i];)

#define vec_loadu_maskedz_f32_32(ptr, mask)                vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((float *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f32_16(ptr, mask)                vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((float *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f32_8(ptr, mask)                 vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((float *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f32_4(ptr, mask)                 vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((float *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f32_1(ptr, mask)                 vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((float *) (ptr))[_i] : 0;)

#define vec_storeu_f32_32(ptr, vec)                        vec_loop_stmt(32, _i, ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f32_16(ptr, vec)                        vec_loop_stmt(16, _i, ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f32_8(ptr, vec)                         vec_loop_stmt( 8, _i, ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f32_4(ptr, vec)                         vec_loop_stmt( 4, _i, ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f32_1(ptr, vec)                         vec_loop_stmt( 1, _i, ((float *) (ptr))[_i] = vec.s[_i];)

#define vec_storeu_masked_f32_32(ptr, vec, mask)           vec_loop_stmt(32, _i, if ((mask).v & (1 << _i)) ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f32_16(ptr, vec, mask)           vec_loop_stmt(16, _i, if ((mask).v & (1 << _i)) ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f32_8(ptr, vec, mask)            vec_loop_stmt( 8, _i, if ((mask).v & (1 << _i)) ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f32_4(ptr, vec, mask)            vec_loop_stmt( 4, _i, if ((mask).v & (1 << _i)) ((float *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f32_1(ptr, vec, mask)            vec_loop_stmt( 1, _i, if ((mask).v & (1 << _i)) ((float *) (ptr))[_i] = vec.s[_i];)


#define vec_gather_f32_i32_32(ptr, idx)                    vec_loop_expr(vec_f32_32_t, 32, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i32_16(ptr, idx)                    vec_loop_expr(vec_f32_16_t, 16, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i32_8(ptr, idx)                     vec_loop_expr(vec_f32_8_t,   8, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i32_4(ptr, idx)                     vec_loop_expr(vec_f32_4_t,   4, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i32_1(ptr, idx)                     vec_loop_expr(vec_f32_1_t,   1, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)

#define vec_gather_f32_i64_32(ptr, idx)                    vec_loop_expr(vec_f32_32_t, 32, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i64_16(ptr, idx)                    vec_loop_expr(vec_f32_16_t, 16, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i64_8(ptr, idx)                     vec_loop_expr(vec_f32_8_t,   8, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i64_4(ptr, idx)                     vec_loop_expr(vec_f32_4_t,   4, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)
#define vec_gather_f32_i64_1(ptr, idx)                     vec_loop_expr(vec_f32_1_t,   1, _tmp, iter, _tmp.s[iter] = (((float *) (ptr))[idx.s[iter]]);)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f32_4(a, b)                                vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f32_1(a, b)                                vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)

#define vec_sub_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f32_4(a, b)                                vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f32_1(a, b)                                vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)

#define vec_mul_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f32_4(a, b)                                vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f32_1(a, b)                                vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)

#define vec_div_f32_32(a, b)                               vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f32_16(a, b)                               vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f32_8(a, b)                                vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f32_4(a, b)                                vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f32_1(a, b)                                vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)

#define vec_fmadd_f32_32(a, b, c)                          vec_loop_expr(vec_f32_32_t, 32, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f32_16(a, b, c)                          vec_loop_expr(vec_f32_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f32_8(a, b, c)                           vec_loop_expr(vec_f32_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f32_4(a, b, c)                           vec_loop_expr(vec_f32_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f32_1(a, b, c)                           vec_loop_expr(vec_f32_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)


#define vec_reduce_add_f32_32(a)                           vec_loop_expr_init(float, 32, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f32_16(a)                           vec_loop_expr_init(float, 16, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f32_8(a)                            vec_loop_expr_init(float,  8, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f32_4(a)                            vec_loop_expr_init(float,  4, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f32_1(a)                            vec_loop_expr_init(float,  1, _tmp, 0, _i, _tmp += a.s[_i];)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cmpeq_f32_32(a, b)                             vec_loop_expr_init(vec_mask_m32_32_t, 32, _tmp, vec_mask_m32_32(0), _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f32_16(a, b)                             vec_loop_expr_init(vec_mask_m32_16_t, 16, _tmp, vec_mask_m32_16(0), _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f32_8(a, b)                              vec_loop_expr_init(vec_mask_m32_8_t,   8, _tmp, vec_mask_m32_8(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f32_4(a, b)                              vec_loop_expr_init(vec_mask_m32_4_t,   4, _tmp, vec_mask_m32_4(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f32_1(a, b)                              vec_loop_expr_init(vec_mask_m32_1_t,   1, _tmp, vec_mask_m32_1(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)

#define vec_cmpgt_f32_32(a, b)                             vec_loop_expr_init(vec_mask_m32_32_t, 32, _tmp, vec_mask_m32_32(0), _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f32_16(a, b)                             vec_loop_expr_init(vec_mask_m32_16_t, 16, _tmp, vec_mask_m32_16(0), _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f32_8(a, b)                              vec_loop_expr_init(vec_mask_m32_8_t,   8, _tmp, vec_mask_m32_8(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f32_4(a, b)                              vec_loop_expr_init(vec_mask_m32_4_t,   4, _tmp, vec_mask_m32_4(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f32_1(a, b)                              vec_loop_expr_init(vec_mask_m32_1_t,   1, _tmp, vec_mask_m32_1(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)


#endif /* VECTORIZATION_SCALAR_F32_H */


