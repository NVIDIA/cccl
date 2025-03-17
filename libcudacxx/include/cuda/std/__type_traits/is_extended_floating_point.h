//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_extended_floating_point : false_type
{};

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v
#  if defined(_CCCL_NO_INLINE_VARIABLES)
  = __is_extended_floating_point<_Tp>::value;
#  else // ^^^ _CCCL_NO_INLINE_VARIABLES ^^^ / vvv !_CCCL_NO_INLINE_VARIABLES vvv
  = false;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

#if _CCCL_HAS_NVFP16()
template <>
struct __is_extended_floating_point<__half> : true_type
{};

#  ifndef _CCCL_NO_INLINE_VARIABLES
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__half> = true;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()
template <>
struct __is_extended_floating_point<__nv_bfloat16> : true_type
{};

#  ifndef _CCCL_NO_INLINE_VARIABLES
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_bfloat16> = true;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
template <>
struct __is_extended_floating_point<__nv_fp8_e4m3> : true_type
{};
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
template <>
struct __is_extended_floating_point<__nv_fp8_e5m2> : true_type
{};
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
template <>
struct __is_extended_floating_point<__nv_fp8_e8m0> : true_type
{};
#endif // _CCCL_HAS_NVFP8_E8M0()

#ifndef _CCCL_NO_INLINE_VARIABLES
#  if _CCCL_HAS_NVFP8_E4M3()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp8_e4m3> = true;
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp8_e5m2> = true;
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp8_e8m0> = true;
#  endif // _CCCL_HAS_NVFP8_E8M0()
#endif // !_CCCL_NO_INLINE_VARIABLES

#if _CCCL_HAS_NVFP6_E2M3()
template <>
struct __is_extended_floating_point<__nv_fp6_e2m3> : true_type
{};
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
template <>
struct __is_extended_floating_point<__nv_fp6_e3m2> : true_type
{};
#endif // _CCCL_HAS_NVFP6_E3M2()
#ifndef _CCCL_NO_INLINE_VARIABLES
#  if _CCCL_HAS_NVFP6_E2M3()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp6_e2m3> = true;
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp6_e3m2> = true;
#  endif // _CCCL_HAS_NVFP6_E3M2()
#endif // !_CCCL_NO_INLINE_VARIABLES

#if _CCCL_HAS_NVFP4_E2M1()
template <>
struct __is_extended_floating_point<__nv_fp4_e2m1> : true_type
{};
#endif // _CCCL_HAS_NVFP4_E2M1()
#ifndef _CCCL_NO_INLINE_VARIABLES
#  if _CCCL_HAS_NVFP4_E2M1()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp4_e2m1> = true;
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // !_CCCL_NO_INLINE_VARIABLES

#if _CCCL_HAS_FLOAT128()
template <>
struct __is_extended_floating_point<__float128> : true_type
{};
#endif // _CCCL_HAS_FLOAT128()
#ifndef _CCCL_NO_INLINE_VARIABLES
#  if _CCCL_HAS_FLOAT128()
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__float128> = true;
#  endif // _CCCL_HAS_FLOAT128()
#endif // !_CCCL_NO_INLINE_VARIABLES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
