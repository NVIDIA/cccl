//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_VOID_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_VOID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_VOID) && !defined(_LIBCUDACXX_USE_IS_VOID_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_void : integral_constant<bool, _CCCL_BUILTIN_IS_VOID(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_void_v = __is_void(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_void : public is_same<remove_cv_t<_Tp>, void>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_void_v = is_void<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // defined(_CCCL_BUILTIN_IS_VOID) && !defined(_LIBCUDACXX_USE_IS_VOID_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_VOID_H
