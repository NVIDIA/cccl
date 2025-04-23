//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_BUILTIN_H
#define __CCCL_BUILTIN_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/extended_data_types.h>

//! This file consolidates all compiler builtin detection for CCCL.
//!
//! To work around older compilers not supporting `__has_builtin` we use `_CCCL_CHECK_BUILTIN` that detects more
//! cases
//!
//! * We work around old clang versions (before clang-10) not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * We work around old intel versions (before 2021.3)   not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * We work around old nvhpc versions (before 2022.11)  not supporting __has_builtin via _CCCL_CHECK_BUILTIN
//! * MSVC needs manual handling, has no real way of checking builtins so all is manual
//! * GCC  needs manual handling, before gcc-10 as that finally supports __has_builtin
//!
//! In case compiler support for a builtin is advertised but leads to regressions we explicitly #undef the macro
//!
//! Finally, because `_CCCL_CHECK_BUILTIN` may lead to false positives, we move detection of new builtins over towards
//! just using _CCCL_HAS_BUILTIN

#ifdef __has_builtin
#  define _CCCL_HAS_BUILTIN(__x) __has_builtin(__x)
#else // ^^^ __has_builtin ^^^ / vvv !__has_builtin vvv
#  define _CCCL_HAS_BUILTIN(__x) 0
#endif // !__has_builtin

#ifdef __has_feature
#  define _CCCL_HAS_FEATURE(__x) __has_feature(__x)
#else // ^^^ __has_feature ^^^ / vvv !__has_feature vvv
#  define _CCCL_HAS_FEATURE(__x) 0
#endif // !__has_feature

// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by the compiler and '1' otherwise.
#ifdef __is_identifier
#  define _CCCL_IS_IDENTIFIER(__x) __is_identifier(__x)
#else // ^^^ __is_identifier ^^^ / vvv !__is_identifier vvv
#  define _CCCL_IS_IDENTIFIER(__x) 1
#endif // !__is_identifier

#define _CCCL_HAS_KEYWORD(__x) !(_CCCL_IS_IDENTIFIER(__x))

// https://bugs.llvm.org/show_bug.cgi?id=44517
#define _CCCL_CHECK_BUILTIN(__x) (_CCCL_HAS_BUILTIN(__##__x) || _CCCL_HAS_KEYWORD(__##__x) || _CCCL_HAS_FEATURE(__x))

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_lvalue_reference) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_ADD_LVALUE_REFERENCE(...) __add_lvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_lvalue_reference)

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_pointer) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_ADD_POINTER(...) __add_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_pointer)

// NVCC has issues with function pointers
#if _CCCL_HAS_BUILTIN(__add_rvalue_reference) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_ADD_RVALUE_REFERENCE(...) __add_rvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__add_rvalue_reference)

// TODO: Enable using the builtin __array_rank when https://llvm.org/PR57133 is resolved
#if 0 // _CCCL_CHECK_BUILTIN(array_rank)
#  define _CCCL_BUILTIN_ARRAY_RANK(...) __array_rank(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(array_rank)

#if _CCCL_HAS_BUILTIN(__array_extent)
#  define _CCCL_BUILTIN_ARRAY_EXTENT(...) __array_extent(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__array_extent)

#if _CCCL_CHECK_BUILTIN(builtin_acos) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ACOSF(...) __builtin_acosf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOS(...)  __builtin_acos(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSL(...) __builtin_acosl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_acos)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ACOSF
#  undef _CCCL_BUILTIN_ACOS
#  undef _CCCL_BUILTIN_ACOSL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_acosh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ACOSHF(...) __builtin_acoshf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSH(...)  __builtin_acosh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSHL(...) __builtin_acoshl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_acosh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ACOSHF
#  undef _CCCL_BUILTIN_ACOSH
#  undef _CCCL_BUILTIN_ACOSHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

// nvhpc has a bug where it supports __builtin_addressof but does not mark it via _CCCL_CHECK_BUILTIN
#if _CCCL_CHECK_BUILTIN(builtin_addressof) || _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_ADDRESSOF(...) __builtin_addressof(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_addressof)

#if _CCCL_CHECK_BUILTIN(builtin_asin) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ASINF(...) __builtin_asinf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASIN(...)  __builtin_asin(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINL(...) __builtin_asinl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_asin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ASINF
#  undef _CCCL_BUILTIN_ASIN
#  undef _CCCL_BUILTIN_ASINL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_asinh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ASINHF(...) __builtin_asinhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINH(...)  __builtin_asinh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINHL(...) __builtin_asinhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_asin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ASINHF
#  undef _CCCL_BUILTIN_ASINH
#  undef _CCCL_BUILTIN_ASINHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_assume) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_ASSUME(...) __builtin_assume(__VA_ARGS__)
#elif _CCCL_COMPILER(GCC, >=, 13)
#  define _CCCL_BUILTIN_ASSUME(...) __attribute__((__assume__(__VA_ARGS__)))
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_ASSUME(...) __assume(__VA_ARGS__)
#else
#  define _CCCL_BUILTIN_ASSUME(...)
#endif // _CCCL_CHECK_BUILTIN(builtin_assume)

#if _CCCL_HAS_BUILTIN(__builtin_assume_aligned) || _CCCL_COMPILER(MSVC, >=, 19, 23) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ASSUME_ALIGNED(...) __builtin_assume_aligned(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_assume_aligned)

#if _CCCL_CHECK_BUILTIN(builtin_atan) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATANF(...) __builtin_atanf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN(...)  __builtin_atan(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANL(...) __builtin_atanl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atan)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATANF
#  undef _CCCL_BUILTIN_ATAN
#  undef _CCCL_BUILTIN_ATANL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_atan2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATAN2F(...) __builtin_atan2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN2(...)  __builtin_atan2(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN2L(...) __builtin_atan2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atan2)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATAN2F
#  undef _CCCL_BUILTIN_ATAN2
#  undef _CCCL_BUILTIN_ATAN2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_atanh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATANHF(...) __builtin_atanhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANH(...)  __builtin_atanh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANHL(...) __builtin_atanhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atanh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATANHF
#  undef _CCCL_BUILTIN_ATANH
#  undef _CCCL_BUILTIN_ATANHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

// MSVC supports __builtin_bit_cast from 19.25 on
#if _CCCL_CHECK_BUILTIN(builtin_bit_cast) || _CCCL_COMPILER(MSVC, >, 19, 25)
#  define _CCCL_BUILTIN_BIT_CAST(...) __builtin_bit_cast(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bit_cast)

#if _CCCL_CHECK_BUILTIN(builtin_popcount) || _CCCL_COMPILER(GCC, <, 10) || _CCCL_COMPILER(CLANG) \
  || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_POPCOUNT(...)   __builtin_popcount(__VA_ARGS__)
#  define _CCCL_BUILTIN_POPCOUNTLL(...) __builtin_popcountll(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_popcount)

#if _CCCL_CHECK_BUILTIN(builtin_popcountg)
#  define _CCCL_BUILTIN_POPCOUNTG(...) __builtin_popcountg(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_popcountg)

// nvcc cannot handle __builtin_popcountg
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_POPCOUNTG
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_clz) || _CCCL_COMPILER(GCC, <, 10) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_CLZ(...)   __builtin_clz(__VA_ARGS__)
#  define _CCCL_BUILTIN_CLZLL(...) __builtin_clzll(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_clz)

#if _CCCL_CHECK_BUILTIN(builtin_clzg)
#  define _CCCL_BUILTIN_CLZG(...) __builtin_clzg(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_clzg)

// nvcc cannot handle __builtin_clzg
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_CLZG
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_ctz) || _CCCL_COMPILER(GCC, <, 10) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_CTZ(...)   __builtin_ctz(__VA_ARGS__)
#  define _CCCL_BUILTIN_CTZLL(...) __builtin_ctzll(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ctz)

#if _CCCL_CHECK_BUILTIN(builtin_ctzg)
#  define _CCCL_BUITLIN_CTZG(...) __builtin_ctzg(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ctzg)

// nvcc cannot handle __builtin_ctzg
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUITLIN_CTZG
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_bswap16) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP16(...) __builtin_bswap16(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap16)

#if _CCCL_CHECK_BUILTIN(builtin_bswap32) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP32(...) __builtin_bswap32(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap32)

#if _CCCL_CHECK_BUILTIN(builtin_bswap64) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP64(...) __builtin_bswap64(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap64)

#if _CCCL_CHECK_BUILTIN(builtin_bswap128) // Only available in GCC >= 11 which supports __has_builtin
#  define _CCCL_BUILTIN_BSWAP128(...) __builtin_bswap128(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap128)

// NVCC cannot handle builtins for bswap
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_BSWAP16
#  undef _CCCL_BUILTIN_BSWAP32
#  undef _CCCL_BUILTIN_BSWAP64
#  undef _CCCL_BUILTIN_BSWAP128
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_bitreverse8) && !_CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_BUILTIN_BITREVERSE8(...) __builtin_bitreverse8(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_bitreverse16) && !_CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_BUILTIN_BITREVERSE16(...) __builtin_bitreverse16(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_bitreverse32) && !_CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_BUILTIN_BITREVERSE32(...) __builtin_bitreverse32(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_bitreverse64) && !_CCCL_HAS_CUDA_COMPILER()
#  define _CCCL_BUILTIN_BITREVERSE64(...) __builtin_bitreverse64(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_cbrt) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_CBRTF(...) __builtin_cbrtf(__VA_ARGS__)
#  define _CCCL_BUILTIN_CBRT(...)  __builtin_cbrt(__VA_ARGS__)
#  define _CCCL_BUILTIN_CBRTL(...) __builtin_cbrtl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_cbrt)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "cbrt"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_CBRTF
#  undef _CCCL_BUILTIN_CBRT
#  undef _CCCL_BUILTIN_CBRTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_ceil) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_CEILF(...) __builtin_ceilf(__VA_ARGS__)
#  define _CCCL_BUILTIN_CEIL(...)  __builtin_ceil(__VA_ARGS__)
#  define _CCCL_BUILTIN_CEILL(...) __builtin_ceill(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ceil)

#if _CCCL_HAS_BUILTIN(__builtin_COLUMN) || _CCCL_COMPILER(MSVC, >=, 19, 27)
#  define _CCCL_BUILTIN_COLUMN() __builtin_COLUMN()
#else // ^^^ _CCCL_HAS_BUILTIN(__builtin_COLUMN) ^^^ / vvv !_CCCL_HAS_BUILTIN(__builtin_COLUMN) vvv
#  define _CCCL_BUILTIN_COLUMN() 0
#endif // !_CCCL_HAS_BUILTIN(__builtin_COLUMN)

#if _CCCL_CHECK_BUILTIN(builtin_constant_p) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_CONSTANT_P(...) __builtin_constant_p(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_constant_p)

#if _CCCL_CHECK_BUILTIN(builtin_copysign) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_COPYSIGNF(...) __builtin_copysignf(__VA_ARGS__)
#  define _CCCL_BUILTIN_COPYSIGN(...)  __builtin_copysign(__VA_ARGS__)
#  define _CCCL_BUILTIN_COPYSIGNL(...) __builtin_copysignl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_copysign)

#if _CCCL_CHECK_BUILTIN(builtin_cos) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_COSF(...) __builtin_cosf(__VA_ARGS__)
#  define _CCCL_BUILTIN_COS(...)  __builtin_cos(__VA_ARGS__)
#  define _CCCL_BUILTIN_COSL(...) __builtin_cosl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_cos)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_COSF
#  undef _CCCL_BUILTIN_COS
#  undef _CCCL_BUILTIN_COSL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_cosh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_COSHF(...) __builtin_coshf(__VA_ARGS__)
#  define _CCCL_BUILTIN_COSH(...)  __builtin_cosh(__VA_ARGS__)
#  define _CCCL_BUILTIN_COSHL(...) __builtin_coshl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_cosh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_COSHF
#  undef _CCCL_BUILTIN_COSH
#  undef _CCCL_BUILTIN_COSHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_exp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXPF(...) __builtin_expf(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP(...)  __builtin_exp(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPL(...) __builtin_expl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_exp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "expf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXPF
#  undef _CCCL_BUILTIN_EXP
#  undef _CCCL_BUILTIN_EXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_exp2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXP2F(...) __builtin_exp2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP2(...)  __builtin_exp2(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP2L(...) __builtin_exp2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_exp2)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "exp2"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXP2F
#  undef _CCCL_BUILTIN_EXP2
#  undef _CCCL_BUILTIN_EXP2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_expm1) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXPM1F(...) __builtin_expm1f(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPM1(...)  __builtin_expm1(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPM1L(...) __builtin_expm1l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_expm1)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "expm1"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXPM1F
#  undef _CCCL_BUILTIN_EXPM1
#  undef _CCCL_BUILTIN_EXPM1L
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_expect) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXPECT(...) __builtin_expect(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_expect)

#if _CCCL_CHECK_BUILTIN(builtin_fabs) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FABSF(...) __builtin_fabsf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABS(...)  __builtin_fabs(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABSL(...) __builtin_fabsl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fabs)

#if _CCCL_CHECK_BUILTIN(builtin_floor) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FLOORF(...) __builtin_floorf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FLOOR(...)  __builtin_floor(__VA_ARGS__)
#  define _CCCL_BUILTIN_FLOORL(...) __builtin_floorl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_floor)

#if _CCCL_CHECK_BUILTIN(builtin_fmax) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMAXF(...) __builtin_fmaxf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMAX(...)  __builtin_fmax(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMAXL(...) __builtin_fmaxl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmax)

#if _CCCL_CHECK_BUILTIN(builtin_fmin) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMINF(...) __builtin_fminf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMIN(...)  __builtin_fmin(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMINL(...) __builtin_fminl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmin)

#if _CCCL_HAS_BUILTIN(__builtin_FILE) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC, >=, 19, 27)
#  define _CCCL_BUILTIN_FILE() __builtin_FILE()
#else // ^^^ _CCCL_HAS_BUILTIN(__builtin_FILE) ^^^ / vvv !_CCCL_HAS_BUILTIN(__builtin_FILE) vvv
#  define _CCCL_BUILTIN_FILE() __FILE__
#endif // !_CCCL_HAS_BUILTIN(__builtin_LINE)

#if _CCCL_CHECK_BUILTIN(builtin_fpclassify) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FPCLASSIFY(...) __builtin_fpclassify(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fpclassify)

// nvcc does not implement __builtin_fpclassify
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_FPCLASSIFY
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_frexp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FREXPF(...) __builtin_frexpf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FREXP(...)  __builtin_frexp(__VA_ARGS__)
#  define _CCCL_BUILTIN_FREXPL(...) __builtin_frexpl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_frexp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "frexp"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_FREXPF
#  undef _CCCL_BUILTIN_FREXP
#  undef _CCCL_BUILTIN_FREXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_HAS_BUILTIN(__builtin_FUNCTION) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC, >=, 19, 27)
#  define _CCCL_BUILTIN_FUNCTION() __builtin_FUNCTION()
#else // ^^^ _CCCL_HAS_BUILTIN(__builtin_FUNCTION) ^^^ / vvv !_CCCL_HAS_BUILTIN(__builtin_FUNCTION) vvv
#  define _CCCL_BUILTIN_FUNCTION() "__builtin_FUNCTION is unsupported"
#endif // !_CCCL_HAS_BUILTIN(__builtin_FUNCTION)

#if _CCCL_CHECK_BUILTIN(builtin_huge_valf) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_HUGE_VALF() __builtin_huge_valf()
#endif // _CCCL_CHECK_BUILTIN(builtin_huge_valf)

#if _CCCL_CHECK_BUILTIN(builtin_huge_val) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_HUGE_VAL() __builtin_huge_val()
#endif // _CCCL_CHECK_BUILTIN(builtin_huge_val)

#if _CCCL_CHECK_BUILTIN(builtin_huge_vall) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_HUGE_VALL() __builtin_huge_vall()
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_HUGE_VALL() static_cast<long double>(__builtin_huge_val())
#endif // _CCCL_CHECK_BUILTIN(builtin_huge_vall)

#if _CCCL_HAS_FLOAT128()
#  if _CCCL_CHECK_BUILTIN(builtin_huge_valf128) || _CCCL_COMPILER(GCC, >=, 7)
#    define _CCCL_BUILTIN_HUGE_VALF128() __builtin_huge_valf128()
#  endif // _CCCL_CHECK_BUILTIN(builtin_huge_valf128) || _CCCL_COMPILER(GCC, >=, 7)

// nvcc does not implement __builtin_huge_valf128
#  if _CCCL_CUDA_COMPILER(NVCC)
#    undef _CCCL_BUILTIN_HUGE_VALF128
#  endif // _CCCL_CUDA_COMPILER(NVCC)
#endif // _CCCL_HAS_FLOAT128()

#if _CCCL_CHECK_BUILTIN(builtin_hypot) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_HYPOTF(...) __builtin_hypotf(__VA_ARGS__)
#  define _CCCL_BUILTIN_HYPOT(...)  __builtin_hypot(__VA_ARGS__)
#  define _CCCL_BUILTIN_HYPOTL(...) __builtin_hypotl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_hypot)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_HYPOTF
#  undef _CCCL_BUILTIN_HYPOT
#  undef _CCCL_BUILTIN_HYPOTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_is_constant_evaluated) || _CCCL_COMPILER(GCC, >=, 9) || _CCCL_COMPILER(MSVC, >, 19, 24)
#  define _CCCL_BUILTIN_IS_CONSTANT_EVALUATED(...) __builtin_is_constant_evaluated(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_is_constant_evaluated)

#if _CCCL_CHECK_BUILTIN(builtin_is_corresponding_member)
#  define _CCCL_BUILTIN_IS_CORRESPONDING_MEMBER(_C1, _C2, _MPtr1, _MPtr2) \
    __builtin_is_corresponding_member(_MPtr1, _MPtr2)
#elif _CCCL_COMPILER(MSVC, >=, 19, 29)
// using __is_corresponding_member with msvc outside of constexpr context causes linker errors, see
// https://developercommunity.visualstudio.com/t/Using-compiler-builtins-causes-linking-n/10888080
// #  define _CCCL_BUILTIN_IS_CORRESPONDING_MEMBER(_C1, _C2, _MPtr1, _MPtr2) __is_corresponding_member(_C1, _C2, _MPtr1,
// _MPtr2)
#endif // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 29) ^^^

#if _CCCL_CHECK_BUILTIN(builtin_is_pointer_interconvertible_with_class)
#  define _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS(_S, _MPtr) \
    __builtin_is_pointer_interconvertible_with_class(_MPtr)
#elif _CCCL_COMPILER(MSVC, >=, 19, 29)
// using __is_pointer_interconvertible_with_class with msvc outside of constexpr context causes linker errors, see
// https://developercommunity.visualstudio.com/t/Using-compiler-builtins-causes-linking-n/10888080
// #  define _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS(_S, _MPtr)
// __is_pointer_interconvertible_with_class(_S, _MPtr)
#endif // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 29) ^^^

#if _CCCL_CHECK_BUILTIN(builtin_isfinite) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(NVRTC, >, 12, 2)
#  define _CCCL_BUILTIN_ISFINITE(...) __builtin_isfinite(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isfinite)

#if _CCCL_CHECK_BUILTIN(builtin_isinf) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISINF(...) __builtin_isinf(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isinf)

#if _CCCL_CHECK_BUILTIN(builtin_isnan) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISNAN(...) __builtin_isnan(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnan)

#if _CCCL_CHECK_BUILTIN(builtin_isnormal) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISNORMAL(...) __builtin_isnormal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnormal)

// nvcc does not implement __builtin_isnormal
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_ISNORMAL
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(builtin_launder) || _CCCL_COMPILER(GCC, >=, 7)
#  define _CCCL_BUILTIN_LAUNDER(...) __builtin_launder(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_launder) && gcc >= 7

#if _CCCL_CHECK_BUILTIN(builtin_ldexp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LDEXPF(...) __builtin_ldexpf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LDEXP(...)  __builtin_ldexp(__VA_ARGS__)
#  define _CCCL_BUILTIN_LDEXPL(...) __builtin_ldexpl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ldexp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "ldexp"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LDEXPF
#  undef _CCCL_BUILTIN_LDEXP
#  undef _CCCL_BUILTIN_LDEXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_lgamma) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LGAMMAF(...) __builtin_lgammaf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LGAMMA(...)  __builtin_lgamma(__VA_ARGS__)
#  define _CCCL_BUILTIN_LGAMMAL(...) __builtin_lgammal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_lgamma)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LGAMMAF
#  undef _CCCL_BUILTIN_LGAMMA
#  undef _CCCL_BUILTIN_LGAMMAL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_HAS_BUILTIN(__builtin_LINE) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC, >=, 19, 27)
#  define _CCCL_BUILTIN_LINE() __builtin_LINE()
#else // ^^^ _CCCL_HAS_BUILTIN(__builtin_LINE) ^^^ / vvv !_CCCL_HAS_BUILTIN(__builtin_LINE) vvv
#  define _CCCL_BUILTIN_LINE() __LINE__
#endif // !_CCCL_HAS_BUILTIN(__builtin_LINE)

#if _CCCL_CHECK_BUILTIN(builtin_llrint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LLRINTF(...) __builtin_llrintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLRINT(...)  __builtin_llrint(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLRINTL(...) __builtin_llrintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_llrint)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "llrint"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LLRINTF
#  undef _CCCL_BUILTIN_LLRINT
#  undef _CCCL_BUILTIN_LLRINTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_llround) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LLROUNDF(...) __builtin_llroundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLROUND(...)  __builtin_llround(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLROUNDL(...) __builtin_llroundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_llround)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "llround"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LLROUNDF
#  undef _CCCL_BUILTIN_LLROUND
#  undef _CCCL_BUILTIN_LLROUNDL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_lrint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LRINTF(...) __builtin_lrintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LRINT(...)  __builtin_lrint(__VA_ARGS__)
#  define _CCCL_BUILTIN_LRINTL(...) __builtin_lrintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_lrint)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "lrint"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LRINTF
#  undef _CCCL_BUILTIN_LRINT
#  undef _CCCL_BUILTIN_LRINTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_lround) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LROUNDF(...) __builtin_lroundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LROUND(...)  __builtin_lround(__VA_ARGS__)
#  define _CCCL_BUILTIN_LROUNDL(...) __builtin_lroundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_lround)

// Below 11.7 nvcc treats the builtin as a host only function
// clang-cuda fails with fatal error: error in backend: Undefined external symbol "lround"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LROUNDF
#  undef _CCCL_BUILTIN_LROUND
#  undef _CCCL_BUILTIN_LROUNDL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_nanf) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NANF(...) __builtin_nanf(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nanf)

#if _CCCL_CHECK_BUILTIN(builtin_nan) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NAN(...) __builtin_nan(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nan)

#if _CCCL_CHECK_BUILTIN(builtin_nanl) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NANL(...) __builtin_nanl(__VA_ARGS__)
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_NANL(...) static_cast<long double>(__builtin_nan(__VA_ARGS__))
#endif // _CCCL_CHECK_BUILTIN(builtin_nanl)

#if _CCCL_HAS_FLOAT128()
#  if _CCCL_CHECK_BUILTIN(builtin_nanf128) || _CCCL_COMPILER(GCC, >=, 7)
#    define _CCCL_BUILTIN_NANF128(...) __builtin_nanf128(__VA_ARGS__)
#  endif // _CCCL_CHECK_BUILTIN(builtin_nanf128) || _CCCL_COMPILER(GCC, >=, 7)

// nvcc does not implement __builtin_nanf128
#  if _CCCL_CUDA_COMPILER(NVCC)
#    undef _CCCL_BUILTIN_NANF128
#  endif // _CCCL_CUDA_COMPILER(NVCC)
#endif // _CCCL_HAS_FLOAT128()

#if _CCCL_CHECK_BUILTIN(builtin_nansf) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NANSF(...) __builtin_nansf(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nansf)

#if _CCCL_CHECK_BUILTIN(builtin_nans) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NANS(...) __builtin_nans(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nans)

#if _CCCL_CHECK_BUILTIN(builtin_nansl) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_NANSL(...) __builtin_nansl(__VA_ARGS__)
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_NANSL(...) static_cast<long double>(__builtin_nans(__VA_ARGS__))
#endif // _CCCL_CHECK_BUILTIN(builtin_nansl)

#if _CCCL_HAS_FLOAT128()
#  if _CCCL_CHECK_BUILTIN(builtin_nansf128) || _CCCL_COMPILER(GCC, >=, 7)
#    define _CCCL_BUILTIN_NANSF128(...) __builtin_nansf128(__VA_ARGS__)
#  endif // _CCCL_CHECK_BUILTIN(builtin_nansf128) || _CCCL_COMPILER(GCC, >=, 7)

// nvcc does not implement __builtin_nansf128
#  if _CCCL_CUDA_COMPILER(NVCC)
#    undef _CCCL_BUILTIN_NANSF128
#  endif // _CCCL_CUDA_COMPILER(NVCC)
#endif // _CCCL_HAS_FLOAT128()

#if _CCCL_CHECK_BUILTIN(builtin_nearbyint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEARBYINTF(...) __builtin_nearbyintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEARBYINT(...)  __builtin_nearbyint(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEARBYINTL(...) __builtin_nearbyintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nearbyint)

#if _CCCL_CHECK_BUILTIN(builtin_nextafter) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEXTAFTERF(...) __builtin_nextafterf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTAFTER(...)  __builtin_nextafter(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTAFTERL(...) __builtin_nextafterl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nextafter)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "nextafter"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_NEXTAFTERF
#  undef _CCCL_BUILTIN_NEXTAFTER
#  undef _CCCL_BUILTIN_NEXTAFTERL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_nexttoward) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEXTTOWARDF(...) __builtin_nexttowardf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTTOWARD(...)  __builtin_nexttoward(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTTOWARDL(...) __builtin_nexttowardl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nexttoward)

#if _CCCL_CHECK_BUILTIN(builtin_log) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOGF(...) __builtin_logf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG(...)  __builtin_log(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOGL(...) __builtin_logl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "logf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOGF
#  undef _CCCL_BUILTIN_LOG
#  undef _CCCL_BUILTIN_LOGL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_log10) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG10F(...) __builtin_log10f(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG10(...)  __builtin_log10(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG10L(...) __builtin_log10l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log10)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log10f"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG10F
#  undef _CCCL_BUILTIN_LOG10
#  undef _CCCL_BUILTIN_LOG10L
#endif //  _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_ilogb) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ILOGBF(...) __builtin_ilogbf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ILOGB(...)  __builtin_ilogb(__VA_ARGS__)
#  define _CCCL_BUILTIN_ILOGBL(...) __builtin_ilogbl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log10)

// Below 11.7 nvcc treats the builtin as a host only function
// clang-cuda fails with fatal error: error in backend: Undefined external symbol "ilogb"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ILOGBF
#  undef _CCCL_BUILTIN_ILOGB
#  undef _CCCL_BUILTIN_ILOGBL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_log1p) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG1PF(...) __builtin_log1pf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG1P(...)  __builtin_log1p(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG1PL(...) __builtin_log1pl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log1p)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log1p"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG1PF
#  undef _CCCL_BUILTIN_LOG1P
#  undef _CCCL_BUILTIN_LOG1PL
#endif //  _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_log2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG2F(...) __builtin_log2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG2(...)  __builtin_log2(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG2L(...) __builtin_log2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log1)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log2f"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG2F
#  undef _CCCL_BUILTIN_LOG2
#  undef _CCCL_BUILTIN_LOG2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_logb) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOGBF(...) __builtin_logbf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOGB(...)  __builtin_logb(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOGBL(...) __builtin_logbl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log1)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "logb"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOGBF
#  undef _CCCL_BUILTIN_LOGB
#  undef _CCCL_BUILTIN_LOGBL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_memcmp) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC, >=, 19, 28)
#  define _CCCL_BUILTIN_MEMCMP(...) __builtin_memcmp(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_memcmp) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC, >=, 19, 28)

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_MEMCMP
#endif // _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_memmove) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_MEMMOVE(...) __builtin_memmove(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_memmove) || _CCCL_COMPILER(GCC)

#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_MEMMOVE
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(__builtin_operator_new) && _CCCL_CHECK_BUILTIN(__builtin_operator_delete) \
  && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_OPERATOR_DELETE(...) __builtin_operator_delete(__VA_ARGS__)
#  define _CCCL_BUILTIN_OPERATOR_NEW(...)    __builtin_operator_new(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(__builtin_operator_new) && _CCCL_CHECK_BUILTIN(__builtin_operator_delete)

#if _CCCL_CHECK_BUILTIN(builtin_pow) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_POWF(...) __builtin_powf(__VA_ARGS__)
#  define _CCCL_BUILTIN_POW(...)  __builtin_pow(__VA_ARGS__)
#  define _CCCL_BUILTIN_POWL(...) __builtin_powl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_pow)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "pow"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_POWF
#  undef _CCCL_BUILTIN_POW
#  undef _CCCL_BUILTIN_POWL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_prefetch) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_PREFETCH(...) NV_IF_TARGET(NV_IS_HOST, __builtin_prefetch(__VA_ARGS__);)
#else
#  define _CCCL_BUILTIN_PREFETCH(...)
#endif // _CCCL_CHECK_BUILTIN(builtin_prefetch)

#if _CCCL_CHECK_BUILTIN(builtin_rint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_RINTF(...) __builtin_rintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_RINT(...)  __builtin_rint(__VA_ARGS__)
#  define _CCCL_BUILTIN_RINTL(...) __builtin_rintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_rint)

#if _CCCL_CHECK_BUILTIN(builtin_round) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ROUNDF(...) __builtin_roundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ROUND(...)  __builtin_round(__VA_ARGS__)
#  define _CCCL_BUILTIN_ROUNDL(...) __builtin_roundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_round)

#if _CCCL_CHECK_BUILTIN(builtin_scalbln) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SCALBLNF(...) __builtin_scalblnf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBLN(...)  __builtin_scalbln(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBLNL(...) __builtin_scalblnl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_scalbln)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "scalblnf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SCALBLNF
#  undef _CCCL_BUILTIN_SCALBLN
#  undef _CCCL_BUILTIN_SCALBLNL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_scalbn) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SCALBNF(...) __builtin_scalbnf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBN(...)  __builtin_scalbn(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBNL(...) __builtin_scalbnl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_scalbn)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "scalbnf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SCALBNF
#  undef _CCCL_BUILTIN_SCALBN
#  undef _CCCL_BUILTIN_SCALBNL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_signbit) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SIGNBIT(...) __builtin_signbit(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_signbit)

#if _CCCL_CHECK_BUILTIN(builtin_sin) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINF(...) __builtin_sinf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SIN(...)  __builtin_sin(__VA_ARGS__)
#  define _CCCL_BUILTIN_SINL(...) __builtin_sinl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SINF
#  undef _CCCL_BUILTIN_SIN
#  undef _CCCL_BUILTIN_SINL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_sinh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINHF(...) __builtin_sinhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SINH(...)  __builtin_sinh(__VA_ARGS__)
#  define _CCCL_BUILTIN_SINHL(...) __builtin_sinhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SINHF
#  undef _CCCL_BUILTIN_SINH
#  undef _CCCL_BUILTIN_SINHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_sqrt) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SQRTF(...) __builtin_sqrtf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SQRT(...)  __builtin_sqrt(__VA_ARGS__)
#  define _CCCL_BUILTIN_SQRTL(...) __builtin_sqrtl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sqrt)

#if _CCCL_CHECK_BUILTIN(builtin_tan) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TANF(...) __builtin_tanf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TAN(...)  __builtin_tan(__VA_ARGS__)
#  define _CCCL_BUILTIN_TANL(...) __builtin_tanl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_tan)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_TANF
#  undef _CCCL_BUILTIN_TAN
#  undef _CCCL_BUILTIN_TANL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_tanh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TANHF(...) __builtin_tanhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TANH(...)  __builtin_tanh(__VA_ARGS__)
#  define _CCCL_BUILTIN_TANHL(...) __builtin_tanhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_tan)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_TANHF
#  undef _CCCL_BUILTIN_TANH
#  undef _CCCL_BUILTIN_TANHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_tgamma) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TGAMMAF(...) __builtin_tgammaf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TGAMMA(...)  __builtin_tgamma(__VA_ARGS__)
#  define _CCCL_BUILTIN_TGAMMAL(...) __builtin_tgammal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_tgamma)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_TGAMMAF
#  undef _CCCL_BUILTIN_TGAMMA
#  undef _CCCL_BUILTIN_TGAMMAL
#endif // _CCCL_CUDA_COMPILER(CLANG)

#if _CCCL_CHECK_BUILTIN(builtin_trunc) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TRUNCF(...) __builtin_truncf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TRUNC(...)  __builtin_trunc(__VA_ARGS__)
#  define _CCCL_BUILTIN_TRUNCL(...) __builtin_truncl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_trunc)

#if _CCCL_HAS_BUILTIN(__decay) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_DECAY(...) __decay(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__decay) && clang-cuda

#if _CCCL_CHECK_BUILTIN(has_nothrow_assign) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_ASSIGN(...) __has_nothrow_assign(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_assign) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_nothrow_constructor) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_CONSTRUCTOR(...) __has_nothrow_constructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_constructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_nothrow_copy) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_NOTHROW_COPY(...) __has_nothrow_copy(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_nothrow_copy) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_trivial_constructor) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_CONSTRUCTOR(...) __has_trivial_constructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_trivial_constructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_trivial_destructor) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(...) __has_trivial_destructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_trivial_destructor) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(has_unique_object_representations) || _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_HAS_UNIQUE_OBJECT_REPRESENTATIONS(...) __has_unique_object_representations(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_unique_object_representations) && gcc >= 7.0

#if _CCCL_CHECK_BUILTIN(has_virtual_destructor) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(...) __has_virtual_destructor(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(has_virtual_destructor) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__integer_pack)
#  define _CCCL_BUILTIN_INTEGER_PACK(...) __integer_pack(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__integer_pack)

#if _CCCL_CHECK_BUILTIN(is_aggregate) || _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(MSVC, >, 19, 14) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_AGGREGATE(...) __is_aggregate(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_aggregate) && gcc >= 7.0

#if _CCCL_CHECK_BUILTIN(is_array)
#  define _CCCL_BUILTIN_IS_ARRAY(...) __is_array(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_array)

// clang prior to clang-19 gives wrong results for __is_array of _Tp[0]
#if _CCCL_COMPILER(CLANG, <, 19)
#  undef _CCCL_BUILTIN_IS_ARRAY
#endif // clang < 19

#if _CCCL_CHECK_BUILTIN(is_assignable) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC, >=, 9)
#  define _CCCL_BUILTIN_IS_ASSIGNABLE(...) __is_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_assignable) && gcc >= 9.0

#if _CCCL_CHECK_BUILTIN(is_base_of) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_BASE_OF(...) __is_base_of(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_base_of) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_class) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_CLASS(...) __is_class(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_class) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__is_compound)
#  define _CCCL_BUILTIN_IS_COMPOUND(...) __is_compound(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_compound)

#if _CCCL_HAS_BUILTIN(__is_const)
#  define _CCCL_BUILTIN_IS_CONST(...) __is_const(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_const)

#if _CCCL_CHECK_BUILTIN(is_constructible) || _CCCL_COMPILER(GCC, >=, 8) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_CONSTRUCTIBLE(...) __is_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_constructible) && gcc >= 8.0

#if _CCCL_CHECK_BUILTIN(is_convertible_to) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_convertible_to)

#if _CCCL_CHECK_BUILTIN(is_destructible) || _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_IS_DESTRUCTIBLE(...) __is_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_destructible)

#if _CCCL_CHECK_BUILTIN(is_empty) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_EMPTY(...) __is_empty(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_empty) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_enum) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_ENUM(...) __is_enum(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_enum) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_final) || _CCCL_COMPILER(GCC, >=, 4, 7) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)

#  define _CCCL_BUILTIN_IS_FINAL(...) __is_final(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_final) && gcc >= 4.7

#if _CCCL_CHECK_BUILTIN(is_function)
#  define _CCCL_BUILTIN_IS_FUNCTION(...) __is_function(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_function)

// All current versions of NVCC give wrong results with __is_function
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_IS_FUNCTION
#endif // _CCCL_CUDA_COMPILER(NVCC)

#if _CCCL_CHECK_BUILTIN(is_fundamental)
#  define _CCCL_BUILTIN_IS_FUNDAMENTAL(...) __is_fundamental(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_fundamental)

// clang prior to clang-10 gives wrong results for __is_fundamental
#if _CCCL_COMPILER(CLANG, <, 10)
#  undef _CCCL_BUILTIN_IS_FUNDAMENTAL
#endif // clang < 10

#if _CCCL_HAS_BUILTIN(__is_integral)
#  define _CCCL_BUILTIN_IS_INTEGRAL(...) __is_integral(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_integral)

#if _CCCL_CHECK_BUILTIN(is_layout_compatible) || _CCCL_COMPILER(MSVC, >=, 19, 29)
#  define _CCCL_BUILTIN_IS_LAYOUT_COMPATIBLE(...) __is_layout_compatible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_layout_compatible) || _CCCL_COMPILER(MSVC, >=, 19, 29)

#if _CCCL_CHECK_BUILTIN(is_literal_type) || _CCCL_COMPILER(GCC, >=, 4, 6) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_LITERAL(...) __is_literal_type(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_literal_type) && gcc >= 4.6

#if _CCCL_CHECK_BUILTIN(is_lvalue_reference)
#  define _CCCL_BUILTIN_IS_LVALUE_REFERENCE(...) __is_lvalue_reference(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_lvalue_reference)

#if _CCCL_HAS_BUILTIN(__is_member_function_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_FUNCTION_POINTER(...) __is_member_function_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_function_pointer)

#if _CCCL_HAS_BUILTIN(__is_member_object_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_OBJECT_POINTER(...) __is_member_object_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_object_pointer)

#if _CCCL_HAS_BUILTIN(__is_member_pointer)
#  define _CCCL_BUILTIN_IS_MEMBER_POINTER(...) __is_member_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_member_pointer)

#if _CCCL_CHECK_BUILTIN(is_nothrow_assignable) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE(...) __is_nothrow_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_assignable)

#if _CCCL_CHECK_BUILTIN(is_nothrow_constructible) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(...) __is_nothrow_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_constructible)

#if _CCCL_CHECK_BUILTIN(is_nothrow_destructible) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(...) __is_nothrow_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_nothrow_destructible)

#if _CCCL_CHECK_BUILTIN(is_object)
#  define _CCCL_BUILTIN_IS_OBJECT(...) __is_object(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_object)

#if _CCCL_CHECK_BUILTIN(is_pod) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_POD(...) __is_pod(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_pod) && gcc >= 4.3

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_pointer)
#  define _CCCL_BUILTIN_IS_POINTER(...) __is_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_pointer)

#if _CCCL_CHECK_BUILTIN(is_pointer_interconvertible_base_of) || _CCCL_COMPILER(MSVC, >=, 19, 29)
#  define _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF(...) __is_pointer_interconvertible_base_of(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_pointer_interconvertible_base_of) || _CCCL_COMPILER(MSVC, >=, 19, 29)

#if _CCCL_CHECK_BUILTIN(is_polymorphic) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_POLYMORPHIC(...) __is_polymorphic(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_polymorphic) && gcc >= 4.3

#if _CCCL_HAS_BUILTIN(__is_reference)
#  define _CCCL_BUILTIN_IS_REFERENCE(...) __is_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_reference)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_referenceable)
#  define _CCCL_BUILTIN_IS_REFERENCEABLE(...) __is_referenceable(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_referenceable)

#if _CCCL_HAS_BUILTIN(__is_rvalue_reference)
#  define _CCCL_BUILTIN_IS_RVALUE_REFERENCE(...) __is_rvalue_reference(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_rvalue_reference)

#if _CCCL_CHECK_BUILTIN(is_same) && !_CCCL_CUDA_COMPILER(NVCC)
#  define _CCCL_BUILTIN_IS_SAME(...) __is_same(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_same)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_scalar)
#  define _CCCL_BUILTIN_IS_SCALAR(...) __is_scalar(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_scalar)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_signed)
#  define _CCCL_BUILTIN_IS_SIGNED(...) __is_signed(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_signed)

#if _CCCL_CHECK_BUILTIN(is_standard_layout) || _CCCL_COMPILER(GCC, >=, 4, 7) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_STANDARD_LAYOUT(...) __is_standard_layout(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_standard_layout) && gcc >= 4.7

#if _CCCL_CHECK_BUILTIN(is_trivial) || _CCCL_COMPILER(GCC, >=, 4, 5) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIAL(...) __is_trivial(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivial) && gcc >= 4.5

#if _CCCL_CHECK_BUILTIN(is_trivially_assignable) || _CCCL_COMPILER(GCC, >=, 5, 1) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(...) __is_trivially_assignable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_assignable) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_constructible) || _CCCL_COMPILER(GCC, >=, 5, 1) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_CONSTRUCTIBLE(...) __is_trivially_constructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_constructible) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_copyable) || _CCCL_COMPILER(GCC, >=, 5, 1) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)

#  define _CCCL_BUILTIN_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_copyable) && gcc >= 5.1

#if _CCCL_CHECK_BUILTIN(is_trivially_destructible) || _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(...) __is_trivially_destructible(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_trivially_destructible)

#if _CCCL_CHECK_BUILTIN(is_union) || _CCCL_COMPILER(GCC, >=, 4, 3) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_UNION(...) __is_union(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_union) && gcc >= 4.3

#if _CCCL_CHECK_BUILTIN(is_unsigned)
#  define _CCCL_BUILTIN_IS_UNSIGNED(...) __is_unsigned(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_unsigned)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_void)
#  define _CCCL_BUILTIN_IS_VOID(...) __is_void(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_void)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__is_volatile)
#  define _CCCL_BUILTIN_IS_VOLATILE(...) __is_volatile(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_volatile)

#if _CCCL_CHECK_BUILTIN(make_integer_seq) || _CCCL_COMPILER(MSVC, >=, 19, 23)
#  define _CCCL_BUILTIN_MAKE_INTEGER_SEQ(...) __make_integer_seq<__VA_ARGS__>
#endif // _CCCL_CHECK_BUILTIN(make_integer_seq)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__make_signed)
#  define _CCCL_BUILTIN_MAKE_SIGNED(...) __make_signed(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__make_signed)

// Disabled due to libstdc++ conflict
#if 0 // _CCCL_HAS_BUILTIN(__make_unsigned)
#  define _CCCL_BUILTIN_MAKE_UNSIGNED(...) __make_unsigned(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__make_unsigned)

#if _CCCL_HAS_BUILTIN(__reference_constructs_from_temporary)
#  define _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY(...) __reference_constructs_from_temporary(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__reference_constructs_from_temporary)

#if _CCCL_HAS_BUILTIN(__reference_converts_from_temporary)
#  define _CCCL_BUILTIN_REFERENCE_CONVERTS_FROM_TEMPORARY(...) __reference_converts_from_temporary(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__reference_converts_from_temporary)

#if _CCCL_HAS_BUILTIN(__remove_all_extents) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_ALL_EXTENTS(...) __remove_all_extents(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_all_extents)

#if _CCCL_HAS_BUILTIN(__remove_const) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_CONST(...) __remove_const(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_const)

#if _CCCL_HAS_BUILTIN(__remove_cv) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_CV(...) __remove_cv(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_cv)

#if _CCCL_HAS_BUILTIN(__remove_cvref) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_CVREF(...) __remove_cvref(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_cvref)

#if _CCCL_COMPILER(NVRTC, <, 12, 4) // NVRTC below 12.4 fails to properly compile that builtin
#  undef _CCCL_BUILTIN_REMOVE_CVREF
#endif // _CCCL_COMPILER(NVRTC, <, 12, 4)

#if _CCCL_HAS_BUILTIN(__remove_extent) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_EXTENT(...) __remove_extent(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_extent)

#if _CCCL_HAS_BUILTIN(__remove_pointer) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_POINTER(...) __remove_pointer(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_pointer)

#if _CCCL_HAS_BUILTIN(__remove_reference)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference(__VA_ARGS__)
#elif _CCCL_HAS_BUILTIN(__remove_reference_t) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference_t(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_reference_t)

#if _CCCL_COMPILER(NVRTC, <, 12, 4) // NVRTC below 12.4 fails to properly compile cuda::std::move with that
#  undef _CCCL_BUILTIN_REMOVE_REFERENCE_T
#endif // _CCCL_COMPILER(NVRTC, <, 12, 4)

#if _CCCL_HAS_BUILTIN(__remove_volatile) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_VOLATILE(...) __remove_volatile(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_volatile)

#if _CCCL_HAS_BUILTIN(__type_pack_element)
#  define _CCCL_BUILTIN_TYPE_PACK_ELEMENT(...) __type_pack_element<__VA_ARGS__>
#endif // _CCCL_HAS_BUILTIN(__type_pack_element)

// NVCC prior to 12.2 have trouble with pack expansion into __type_pack_element in an alias template
#if _CCCL_CUDACC_BELOW(12, 2)
#  undef _CCCL_BUILTIN_TYPE_PACK_ELEMENT
#endif // _CCCL_CUDACC_BELOW(12, 2)

#if _CCCL_CHECK_BUILTIN(underlying_type) || _CCCL_COMPILER(GCC, >=, 4, 7) || _CCCL_COMPILER(MSVC) \
  || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_UNDERLYING_TYPE(...) __underlying_type(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(underlying_type) && gcc >= 4.7

#if _CCCL_COMPILER(MSVC) // To use __builtin_FUNCSIG(), both MSVC and nvcc need to support it
#  if _CCCL_COMPILER(MSVC, >=, 19, 35) && _CCCL_CUDACC_AT_LEAST(12, 3)
#    define _CCCL_BUILTIN_PRETTY_FUNCTION() __builtin_FUNCSIG()
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 35) ^^^ / vvv _CCCL_COMPILER(MSVC, <, 19, 35) vvv
#    define _CCCL_BUILTIN_PRETTY_FUNCTION() __FUNCSIG__
#    define _CCCL_BROKEN_MSVC_FUNCSIG
#  endif // _CCCL_COMPILER(MSVC, <, 19, 35)
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_BUILTIN_PRETTY_FUNCTION() __PRETTY_FUNCTION__
#endif // !_CCCL_COMPILER(MSVC)

// GCC's builtin_strlen isn't reliable at constexpr time
// NVRTC does not expose builtin_strlen
#if !_CCCL_COMPILER(GCC) && !_CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_STRLEN(...) __builtin_strlen(__VA_ARGS__)
#endif

#endif // __CCCL_BUILTIN_H
