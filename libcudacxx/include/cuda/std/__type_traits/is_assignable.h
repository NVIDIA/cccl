//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ASSIGNABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/declval.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename, typename _Tp>
struct __select_2nd
{
  typedef _CCCL_NODEBUG_ALIAS _Tp type;
};

#if defined(_CCCL_BUILTIN_IS_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_ASSIGNABLE_FALLBACK)

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_assignable
    : public integral_constant<bool, _CCCL_BUILTIN_IS_ASSIGNABLE(_T1, _T2)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _T1, class _T2>
_CCCL_INLINE_VAR constexpr bool is_assignable_v = _CCCL_BUILTIN_IS_ASSIGNABLE(_T1, _T2);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else

template <class _Tp, class _Arg>
_LIBCUDACXX_HIDE_FROM_ABI
typename __select_2nd<decltype((_CUDA_VSTD::declval<_Tp>() = _CUDA_VSTD::declval<_Arg>())), true_type>::type
__is_assignable_test(int);

template <class, class>
_LIBCUDACXX_HIDE_FROM_ABI false_type __is_assignable_test(...);

template <class _Tp, class _Arg, bool = is_void<_Tp>::value || is_void<_Arg>::value>
struct __is_assignable_imp : public decltype((_CUDA_VSTD::__is_assignable_test<_Tp, _Arg>(0)))
{};

template <class _Tp, class _Arg>
struct __is_assignable_imp<_Tp, _Arg, true> : public false_type
{};

template <class _Tp, class _Arg>
struct is_assignable : public __is_assignable_imp<_Tp, _Arg>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_CCCL_INLINE_VAR constexpr bool is_assignable_v = is_assignable<_Tp, _Arg>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // defined(_CCCL_BUILTIN_IS_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_ASSIGNABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ASSIGNABLE_H
