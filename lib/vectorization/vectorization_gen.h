#ifndef  VECTORIZATION_GEN_H
#define  VECTORIZATION_GEN_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#define VEC_EXPAND(fun, ...)  fun(__VA_ARGS__)

#define _VEC_CONCAT(a, b)  a ## b
#define _VEC_CONCAT_EXPAND(a, b)  _VEC_CONCAT(a, b)
#define VEC_CONCAT(a, b)  _VEC_CONCAT_EXPAND(a, b)


/* We can force specific vector extensions by defining 'VEC_FORCE' and the vector extensions we want.
 */

#ifndef VEC_FORCE
	#undef VEC_X86_512
	#undef VEC_X86_256
	#undef VEC_X86_128
	#undef VEC_ARM_SVE
	#undef VEC_RISCV_SVV

	#if defined(__AVX512F__)
		/* AVX512 is slow. */
		#define VEC_X86_512
		// #define VEC_X86_256
	#elif defined(__AVX2__)
		#define VEC_X86_256
	#elif defined(__AVX__)
		#define VEC_X86_128
	#elif defined(__ARM_FEATURE_SVE)
		#define VEC_ARM_SVE
	#elif defined(__riscv)
		#define VEC_RISCV_SVV
	#else
		#define VEC_SCALAR
	#endif

#endif


#if defined(VEC_X86_512)
	#include "x86/avx512/vectorization_x86_avx512_i32.h"
	#include "x86/avx512/vectorization_x86_avx512_i64.h"
	#include "x86/avx512/vectorization_x86_avx512_f32.h"
	#include "x86/avx512/vectorization_x86_avx512_f64.h"
#elif defined(VEC_X86_256)
	#include "x86/avx256/vectorization_x86_avx256_i32.h"
	#include "x86/avx256/vectorization_x86_avx256_i64.h"
	#include "x86/avx256/vectorization_x86_avx256_f32.h"
	#include "x86/avx256/vectorization_x86_avx256_f64.h"
#elif defined(VEC_X86_128)
	#include "x86/avx128/vectorization_x86_avx128_i32.h"
	#include "x86/avx128/vectorization_x86_avx128_i64.h"
	#include "x86/avx128/vectorization_x86_avx128_f32.h"
	#include "x86/avx128/vectorization_x86_avx128_f64.h"
#elif defined(VEC_ARM_SVE)
	#include "arm/sve/vectorization_arm_sve_i32.h"
	#include "arm/sve/vectorization_arm_sve_i64.h"
	#include "arm/sve/vectorization_arm_sve_f32.h"
	#include "arm/sve/vectorization_arm_sve_f64.h"
#elif defined(VEC_RISCV_SVV)
	#include "riscv/svv/vectorization_riscv_svv_i32.h"
	#include "riscv/svv/vectorization_riscv_svv_i64.h"
	#include "riscv/svv/vectorization_riscv_svv_f32.h"
	#include "riscv/svv/vectorization_riscv_svv_f64.h"
#else
	#include "scalar/vectorization_scalar_i32.h"
	#include "scalar/vectorization_scalar_i64.h"
	#include "scalar/vectorization_scalar_f32.h"
	#include "scalar/vectorization_scalar_f64.h"
#endif


#define VEC_LEN_DEFAULT(suffix)                        VEC_CONCAT(vec_len_default_, suffix)

#define VEC_FORM_NAME(name, suffix, vs)                VEC_CONCAT(VEC_CONCAT(VEC_CONCAT(name, suffix), _), DEFAULT_ARG(VEC_LEN_DEFAULT(suffix), vs))
#define VEC_FORM_NAME_2(name, suffix_1, suffix_2, vs)  VEC_CONCAT(VEC_CONCAT(VEC_CONCAT(VEC_CONCAT(VEC_CONCAT(name, suffix_1), _), suffix_2), _), DEFAULT_ARG(VEC_LEN_DEFAULT(suffix_1), vs))

#define VEC_CALL(name, suffix, vs, ...)                VEC_EXPAND(VEC_FORM_NAME(name, suffix, vs), __VA_ARGS__)
#define VEC_CALL_2(name, suffix_1, suffix_2, vs, ...)  VEC_EXPAND(VEC_FORM_NAME_2(name, suffix_1, suffix_2, vs), __VA_ARGS__)


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                              Generics                                                                  -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


#define vec_t(suffix, vs)                                                      VEC_CONCAT(VEC_FORM_NAME(vec_, suffix, vs), _t)
#define vec_mask_t(suffix, vs)                                                 VEC_CONCAT(VEC_FORM_NAME(vec_mask_, suffix, vs), _t)
#define vec_mask_packed_t(suffix, vs)                                          VEC_CONCAT(VEC_FORM_NAME(vec_mask_packed_, suffix, vs), _t)
#define vec_perm_t(suffix, vs)                                                 VEC_CONCAT(VEC_FORM_NAME(vec_perm_, suffix, vs), _t)
#define vec_perm_elem_t(suffix, vs)                                            VEC_CONCAT(VEC_FORM_NAME(vec_perm_elem_, suffix, vs), _t)

// #define vec_len(suffix, vs)                      VEC_FORM_NAME(vec_len_, suffix, vs)

#define vec_cast_to_i32(suffix, vs, val)                                       VEC_CALL(vec_cast_to_i32_, suffix, vs, val)
#define vec_cast_to_i64(suffix, vs, val)                                       VEC_CALL(vec_cast_to_i64_, suffix, vs, val)
#define vec_cast_to_f32(suffix, vs, val)                                       VEC_CALL(vec_cast_to_f32_, suffix, vs, val)
#define vec_cast_to_f64(suffix, vs, val)                                       VEC_CALL(vec_cast_to_f64_, suffix, vs, val)

#define vec_elem_get(suffix, vs, vec, index)                                   VEC_CALL(vec_elem_get_, suffix, vs, vec, index)
#define vec_elem_set(suffix, vs, vec, index, val)                              VEC_CALL(vec_elem_set_, suffix, vs, vec, index, val)
#define vec_array(suffix, vs, val)                                             VEC_CALL(vec_array_, suffix, vs, val)

#define vec_set1(suffix, vs, val)                                              VEC_CALL(vec_set1_, suffix, vs, val)

#define vec_set_iter(suffix, vs, iter, expr)                                   VEC_CALL(vec_set_iter_, suffix, vs, iter, expr)


#define vec_loadu(suffix, vs, ptr)                                             VEC_CALL(vec_loadu_, suffix, vs, ptr)

#define vec_loadu_maskedz(suffix, vs, ptr, mask)                               VEC_CALL(vec_loadu_maskedz_, suffix, vs, ptr, mask)

#define vec_loadu_firstNz(suffix, vs, ptr, N)                                  VEC_CALL(vec_loadu_firstNz_, suffix, vs, ptr, N)

#define vec_storeu(suffix, vs, ptr, vec)                                       VEC_CALL(vec_storeu_, suffix, vs, ptr, vec)

#define vec_storeu_masked(suffix, vs, ptr, vec, mask)                          VEC_CALL(vec_storeu_masked_, suffix, vs, ptr, vec, mask)

#define vec_gather(suffix_val, suffix_idx, vs, ptr, idx)                       VEC_CALL_2(vec_gather_, suffix_val, suffix_idx, vs, ptr, idx)


#define vec_add(suffix, vs, a, b)                                              VEC_CALL(vec_add_, suffix, vs, a, b)

#define vec_sub(suffix, vs, a, b)                                              VEC_CALL(vec_sub_, suffix, vs, a, b)

#define vec_mul(suffix, vs, a, b)                                              VEC_CALL(vec_mul_, suffix, vs, a, b)

#define vec_div(suffix, vs, a, b)                                              VEC_CALL(vec_div_, suffix, vs, a, b)


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                                Masks                                                                   -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


#define vec_mask_pack(suffix, vs, a)                                           VEC_CALL(vec_mask_pack_, suffix, vs, a)
#define vec_mask_packed_get_bit(suffix, vs, a, pos)                            VEC_CALL(vec_mask_packed_get_bit_, suffix, vs, a, pos)

#define vec_mask_loadu_packed(suffix, vs, expr)                                VEC_CALL(vec_mask_loadu_packed_, suffix, vs, expr)

#define vec_mask_whilelt(suffix, vs, i, N)                                     VEC_CALL(vec_mask_whilelt_, suffix, vs, i, N)
#define vec_mask_firstN(suffix, vs, N)                                         VEC_CALL(vec_mask_firstN_, suffix, vs, N)


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                               Compare                                                                  -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


#define vec_cmpeq(suffix, vs, a, b)                                            VEC_CALL(vec_cmpeq_, suffix, vs, a, b)
#define vec_cmpgt(suffix, vs, a, b)                                            VEC_CALL(vec_cmpgt_, suffix, vs, a, b)


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                              Integers                                                                  -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


#define vec_and(suffix, vs, a, b)                                              VEC_CALL(vec_and_, suffix, vs, a, b)

#define vec_or(suffix, vs, a, b)                                               VEC_CALL(vec_or_, suffix, vs, a, b)

#define vec_not(suffix, vs, a)                                                 VEC_CALL(vec_not_, suffix, vs, a)

#define vec_xor(suffix, vs, a, b)                                              VEC_CALL(vec_xor_, suffix, vs, a, b)

#define vec_slli(suffix, vs, a, imm8)                                          VEC_CALL(vec_slli_, suffix, vs, a, imm8)
#define vec_srli(suffix, vs, a, imm8)                                          VEC_CALL(vec_srli_, suffix, vs, a, imm8)
#define vec_srai(suffix, vs, a, imm8)                                          VEC_CALL(vec_srai_, suffix, vs, a, imm8)

#define vec_sllv(suffix, vs, a, count)                                         VEC_CALL(vec_sllv_, suffix, vs, a, count)
#define vec_srlv(suffix, vs, a, count)                                         VEC_CALL(vec_srlv_, suffix, vs, a, count)
#define vec_srav(suffix, vs, a, count)                                         VEC_CALL(vec_srav_, suffix, vs, a, count)

#define vec_permute(suffix, vs, a, idx)                                        VEC_CALL(vec_permute_, suffix, vs, a, idx)


//==========================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------------
//-                                                               Floats                                                                   -
//------------------------------------------------------------------------------------------------------------------------------------------
//==========================================================================================================================================


// Returns a*b + c
#define vec_fmadd(suffix, vs, a, b, c)                                         VEC_CALL(vec_fmadd_, suffix, vs, a, b, c)

#define vec_reduce_add(suffix, vs, a)                                          VEC_CALL(vec_reduce_add_, suffix, vs, a)


#endif /*  VECTORIZATION_GEN_H */

