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

#include <cuda/std/__type_traits/integral_constant.h>

#if defined(_CCCL_HAS_NVFP16)
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16

#if defined(_CCCL_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif // _CCCL_HAS_NVFP8()

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

#if defined(_CCCL_HAS_NVFP16)
template <>
struct __is_extended_floating_point<__half> : true_type
{};

#  ifndef _CCCL_NO_INLINE_VARIABLES
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__half> = true;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // _CCCL_HAS_NVFP16

#if defined(_CCCL_HAS_NVBF16)
template <>
struct __is_extended_floating_point<__nv_bfloat16> : true_type
{};

#  ifndef _CCCL_NO_INLINE_VARIABLES
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_bfloat16> = true;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP8()
template <>
struct __is_extended_floating_point<__nv_fp8_e4m3> : true_type
{};
template <>
struct __is_extended_floating_point<__nv_fp8_e5m2> : true_type
{};

#  ifndef _CCCL_NO_INLINE_VARIABLES
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp8_e4m3> = true;
template <>
_CCCL_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_fp8_e5m2> = true;
#  endif // !_CCCL_NO_INLINE_VARIABLES
#endif // _CCCL_HAS_NVFP8()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
