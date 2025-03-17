//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_EMPTY) && !defined(_LIBCUDACXX_USE_IS_EMPTY_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_empty : public integral_constant<bool, _CCCL_BUILTIN_IS_EMPTY(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_empty_v = _CCCL_BUILTIN_IS_EMPTY(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else

template <class _Tp>
struct __is_empty1 : public _Tp
{
  double __lx;
};

struct __is_empty2
{
  double __lx;
};

template <class _Tp, bool = _CCCL_TRAIT(is_class, _Tp)>
struct __cccl_empty : public integral_constant<bool, sizeof(__is_empty1<_Tp>) == sizeof(__is_empty2)>
{};

template <class _Tp>
struct __cccl_empty<_Tp, false> : public false_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_empty : public __cccl_empty<_Tp>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_empty_v = is_empty<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // defined(_CCCL_BUILTIN_IS_EMPTY) && !defined(_LIBCUDACXX_USE_IS_EMPTY_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H
