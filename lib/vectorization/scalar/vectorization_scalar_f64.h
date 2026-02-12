#ifndef VECTORIZATION_SCALAR_F64_H
#define VECTORIZATION_SCALAR_F64_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"
#include "vectorization_scalar_m64.h"


typedef union __attribute__((packed, aligned(8))) {         double         v;         float         vf32;         int64_t         vi;  double s[ 1]; int64_t si[ 1]; uint64_t su[ 1]; double sf[ 1]; }  vec_f64_1_t;
typedef union __attribute__((packed, aligned(8))) { struct {double a[ 2];} v; struct {float a[ 4];} vf32; struct {int64_t a[ 4];} vi;  double s[ 2]; int64_t si[ 2]; uint64_t su[ 2]; double sf[ 2]; }  vec_f64_2_t;
typedef union __attribute__((packed, aligned(8))) { struct {double a[ 4];} v; struct {float a[ 8];} vf32; struct {int64_t a[ 4];} vi;  double s[ 4]; int64_t si[ 4]; uint64_t su[ 4]; double sf[ 4]; }  vec_f64_4_t;
typedef union __attribute__((packed, aligned(8))) { struct {double a[ 8];} v; struct {float a[16];} vf32; struct {int64_t a[ 4];} vi;  double s[ 8]; int64_t si[ 8]; uint64_t su[ 8]; double sf[ 8]; }  vec_f64_8_t;
typedef union __attribute__((packed, aligned(8))) { struct {double a[16];} v; struct {float a[32];} vf32; struct {int64_t a[ 4];} vi;  double s[16]; int64_t si[16]; uint64_t su[16]; double sf[16]; }  vec_f64_16_t;

#define vec_len_default_f64  4


#define vec_f64_16(val)                                    ( (vec_f64_16_t) { .v = (val) } )
#define vec_f64_8(val)                                     ( (vec_f64_8_t)  { .v = (val) } )
#define vec_f64_4(val)                                     ( (vec_f64_4_t)  { .v = (val) } )
#define vec_f64_2(val)                                     ( (vec_f64_2_t)  { .v = (val) } )
#define vec_f64_1(val)                                     ( (vec_f64_1_t)  { .v = (val) } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Cast
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cast_to_i32_f64_16(val)                        ( (vec_i32_32_t) { .v = (val).v } )
#define vec_cast_to_i32_f64_8(val)                         ( (vec_i32_16_t) { .v = (val).v } )
#define vec_cast_to_i32_f64_4(val)                         ( (vec_i32_8_t)  { .v = (val).v } )
#define vec_cast_to_i32_f64_2(val)                         ( (vec_i32_4_t)  { .v = (val).v } )
#define vec_cast_to_i32_f64_1(val)                         ( (vec_i32_1_t)  { .v = (val).v } )

#define vec_cast_to_i64_f64_16(val)                        ( (vec_i64_16_t) { .v = (val).vi } )
#define vec_cast_to_i64_f64_8(val)                         ( (vec_i64_8_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_4(val)                         ( (vec_i64_4_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_2(val)                         ( (vec_i64_2_t)  { .v = (val).vi } )
#define vec_cast_to_i64_f64_1(val)                         ( (vec_i64_1_t)  { .v = (val).vi } )

#define vec_cast_to_f32_f64_16(val)                        ( (vec_f32_32_t) { .v = (val).vf32 } )
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

#define vec_set1_f64_16(val)                               vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f64_8(val)                                vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f64_4(val)                                vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f64_2(val)                                vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = (val);)
#define vec_set1_f64_1(val)                                vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = (val);)

#define vec_set_iter_f64_16(iter, expr)                    vec_loop_expr(vec_f64_16_t, 16, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f64_8(iter, expr)                     vec_loop_expr(vec_f64_8_t,   8, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f64_4(iter, expr)                     vec_loop_expr(vec_f64_4_t,   4, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f64_2(iter, expr)                     vec_loop_expr(vec_f64_2_t,   2, _tmp, iter, _tmp.s[iter] = (expr);)
#define vec_set_iter_f64_1(iter, expr)                     vec_loop_expr(vec_f64_1_t,   1, _tmp, iter, _tmp.s[iter] = (expr);)

#define vec_loadu_f64_16(ptr)                              vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = ((double *) (ptr))[_i];)
#define vec_loadu_f64_8(ptr)                               vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = ((double *) (ptr))[_i];)
#define vec_loadu_f64_4(ptr)                               vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = ((double *) (ptr))[_i];)
#define vec_loadu_f64_2(ptr)                               vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = ((double *) (ptr))[_i];)
#define vec_loadu_f64_1(ptr)                               vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = ((double *) (ptr))[_i];)

#define vec_loadu_maskedz_f64_16(ptr, mask)                vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((double *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f64_8(ptr, mask)                 vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((double *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f64_4(ptr, mask)                 vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((double *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f64_2(ptr, mask)                 vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((double *) (ptr))[_i] : 0;)
#define vec_loadu_maskedz_f64_1(ptr, mask)                 vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = ((mask).v & (1 << _i)) ? ((double *) (ptr))[_i] : 0;)

#define vec_storeu_f64_16(ptr, vec)                        vec_loop_stmt(16, _i, ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f64_8(ptr, vec)                         vec_loop_stmt( 8, _i, ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f64_4(ptr, vec)                         vec_loop_stmt( 4, _i, ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f64_2(ptr, vec)                         vec_loop_stmt( 2, _i, ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_f64_1(ptr, vec)                         vec_loop_stmt( 1, _i, ((double *) (ptr))[_i] = vec.s[_i];)

#define vec_storeu_masked_f64_16(ptr, vec, mask)           vec_loop_stmt(16, _i, if ((mask).v & (1 << _i)) ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f64_8(ptr, vec, mask)            vec_loop_stmt( 8, _i, if ((mask).v & (1 << _i)) ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f64_4(ptr, vec, mask)            vec_loop_stmt( 4, _i, if ((mask).v & (1 << _i)) ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f64_2(ptr, vec, mask)            vec_loop_stmt( 2, _i, if ((mask).v & (1 << _i)) ((double *) (ptr))[_i] = vec.s[_i];)
#define vec_storeu_masked_f64_1(ptr, vec, mask)            vec_loop_stmt( 1, _i, if ((mask).v & (1 << _i)) ((double *) (ptr))[_i] = vec.s[_i];)

#define vec_gather_f64_i32_16(ptr, idx)                    vec_loop_expr(vec_f64_16_t, 16, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i32_8(ptr, idx)                     vec_loop_expr(vec_f64_8_t,   8, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i32_4(ptr, idx)                     vec_loop_expr(vec_f64_4_t,   4, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i32_2(ptr, idx)                     vec_loop_expr(vec_f64_2_t,   2, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i32_1(ptr, idx)                     vec_loop_expr(vec_f64_1_t,   1, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)

#define vec_gather_f64_i64_16(ptr, idx)                    vec_loop_expr(vec_f64_16_t, 16, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i64_8(ptr, idx)                     vec_loop_expr(vec_f64_8_t,   8, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i64_4(ptr, idx)                     vec_loop_expr(vec_f64_4_t,   4, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i64_2(ptr, idx)                     vec_loop_expr(vec_f64_2_t,   2, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)
#define vec_gather_f64_i64_1(ptr, idx)                     vec_loop_expr(vec_f64_1_t,   1, _tmp, iter, _tmp.s[iter] = (((double *) (ptr))[idx.s[iter]]);)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_add_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f64_2(a, b)                                vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)
#define vec_add_f64_1(a, b)                                vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] + b.s[_i];)

#define vec_sub_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f64_2(a, b)                                vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)
#define vec_sub_f64_1(a, b)                                vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] - b.s[_i];)

#define vec_mul_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f64_2(a, b)                                vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)
#define vec_mul_f64_1(a, b)                                vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i];)

#define vec_div_f64_16(a, b)                               vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f64_8(a, b)                                vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f64_4(a, b)                                vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f64_2(a, b)                                vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)
#define vec_div_f64_1(a, b)                                vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] / b.s[_i];)

#define vec_fmadd_f64_16(a, b, c)                          vec_loop_expr(vec_f64_16_t, 16, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f64_8(a, b, c)                           vec_loop_expr(vec_f64_8_t,   8, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f64_4(a, b, c)                           vec_loop_expr(vec_f64_4_t,   4, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f64_2(a, b, c)                           vec_loop_expr(vec_f64_2_t,   2, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)
#define vec_fmadd_f64_1(a, b, c)                           vec_loop_expr(vec_f64_1_t,   1, _tmp, _i, _tmp.s[_i] = a.s[_i] * b.s[_i] + c.s[_i];)


#define vec_reduce_add_f64_16(a)                           vec_loop_expr_init(double, 16, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f64_8(a)                            vec_loop_expr_init(double,  8, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f64_4(a)                            vec_loop_expr_init(double,  4, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f64_2(a)                            vec_loop_expr_init(double,  2, _tmp, 0, _i, _tmp += a.s[_i];)
#define vec_reduce_add_f64_1(a)                            vec_loop_expr_init(double,  1, _tmp, 0, _i, _tmp += a.s[_i];)


//------------------------------------------------------------------------------------------------------------------------------------------
//- Compare
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_cmpeq_f64_16(a, b)                             vec_loop_expr_init(vec_mask_m64_16_t, 16, _tmp, vec_mask_m64_16(0), _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f64_8(a, b)                              vec_loop_expr_init(vec_mask_m64_8_t,   8, _tmp, vec_mask_m64_8(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f64_4(a, b)                              vec_loop_expr_init(vec_mask_m64_4_t,   4, _tmp, vec_mask_m64_4(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f64_2(a, b)                              vec_loop_expr_init(vec_mask_m64_2_t,   2, _tmp, vec_mask_m64_2(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)
#define vec_cmpeq_f64_1(a, b)                              vec_loop_expr_init(vec_mask_m64_1_t,   1, _tmp, vec_mask_m64_1(0),  _i, (_tmp).v |= (a.s[_i] == b.s[_i]) << _i;)

#define vec_cmpgt_f64_16(a, b)                             vec_loop_expr_init(vec_mask_m64_16_t, 16, _tmp, vec_mask_m64_16(0), _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f64_8(a, b)                              vec_loop_expr_init(vec_mask_m64_8_t,   8, _tmp, vec_mask_m64_8(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f64_4(a, b)                              vec_loop_expr_init(vec_mask_m64_4_t,   4, _tmp, vec_mask_m64_4(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f64_2(a, b)                              vec_loop_expr_init(vec_mask_m64_2_t,   2, _tmp, vec_mask_m64_2(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)
#define vec_cmpgt_f64_1(a, b)                              vec_loop_expr_init(vec_mask_m64_1_t,   1, _tmp, vec_mask_m64_1(0),  _i, (_tmp).v |= (a.s[_i]  > b.s[_i]) << _i;)


#endif /* VECTORIZATION_SCALAR_F64_H */

