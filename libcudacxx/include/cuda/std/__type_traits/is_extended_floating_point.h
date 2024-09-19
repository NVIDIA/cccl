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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_extended_floating_point : false_type
{};

#if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v = false;
#elif _CCCL_STD_VER >= 2014 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v = __is_extended_floating_point<_Tp>::value;
#endif // _CCCL_STD_VER >= 2014

#if defined(_LIBCUDACXX_HAS_NVFP16)
#  include <cuda_fp16.h>

template <>
struct __is_extended_floating_point<__half> : true_type
{};

#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v<__half> = true;
#  endif // _CCCL_STD_VER >= 2014
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP

template <>
struct __is_extended_floating_point<__nv_bfloat16> : true_type
{};

#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v<__nv_bfloat16> = true;
#  endif // _CCCL_STD_VER >= 2014
#endif // _LIBCUDACXX_HAS_NVBF16

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
