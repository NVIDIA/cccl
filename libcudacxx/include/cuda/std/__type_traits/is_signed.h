//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4197) //  top-level volatile in cast is ignored

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_SIGNED) && !defined(_LIBCUDACXX_USE_IS_SIGNED_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_signed : public integral_constant<bool, _CCCL_BUILTIN_IS_SIGNED(_Tp)>
{};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_signed_v = _CCCL_BUILTIN_IS_SIGNED(_Tp);
#  endif

#else

template <class _Tp, bool = is_integral<_Tp>::value>
struct __libcpp_is_signed_impl : public bool_constant<(_Tp(-1) < _Tp(0))>
{};

template <class _Tp>
struct __libcpp_is_signed_impl<_Tp, false> : public true_type
{}; // floating point

template <class _Tp, bool = is_arithmetic<_Tp>::value>
struct __libcpp_is_signed : public __libcpp_is_signed_impl<_Tp>
{};

template <class _Tp>
struct __libcpp_is_signed<_Tp, false> : public false_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_signed : public __libcpp_is_signed<_Tp>
{};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_signed_v = is_signed<_Tp>::value;
#  endif

#endif // defined(_CCCL_BUILTIN_IS_SIGNED) && !defined(_LIBCUDACXX_USE_IS_SIGNED_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SIGNED_H
