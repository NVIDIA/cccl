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

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_void.h"
#include "../__utility/declval.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<typename, typename _Tp> struct __select_2nd { typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type; };

#if defined(_LIBCUDACXX_IS_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_ASSIGNABLE_FALLBACK)

template <class _T1, class _T2> struct _LIBCUDACXX_TEMPLATE_VIS is_assignable
    : public integral_constant<bool, _LIBCUDACXX_IS_ASSIGNABLE(_T1, _T2)>
    {};

#if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _T1, class _T2>
_LIBCUDACXX_INLINE_VAR constexpr bool is_assignable_v = _LIBCUDACXX_IS_ASSIGNABLE(_T1, _T2);
#endif

#else

template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VISIBILITY
typename __select_2nd<decltype((_CUDA_VSTD::declval<_Tp>() = _CUDA_VSTD::declval<_Arg>())), true_type>::type
__is_assignable_test(int);

template <class, class>
_LIBCUDACXX_INLINE_VISIBILITY
false_type __is_assignable_test(...);

template <class _Tp, class _Arg, bool = is_void<_Tp>::value || is_void<_Arg>::value>
struct __is_assignable_imp
    : public decltype((_CUDA_VSTD::__is_assignable_test<_Tp, _Arg>(0))) {};

template <class _Tp, class _Arg>
struct __is_assignable_imp<_Tp, _Arg, true>
    : public false_type
{
};

template <class _Tp, class _Arg>
struct is_assignable
    : public __is_assignable_imp<_Tp, _Arg> {};

#if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_assignable_v = is_assignable<_Tp, _Arg>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_ASSIGNABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ASSIGNABLE_H
