//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#else
#include "../__type_traits/integral_constant.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_lvalue_reference) && \
    __has_builtin(__is_rvalue_reference) && \
    __has_builtin(__is_reference)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_lvalue_reference : _BoolConstant<__is_lvalue_reference(_Tp)> { };

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_rvalue_reference : _BoolConstant<__is_rvalue_reference(_Tp)> { };

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_reference : _BoolConstant<__is_reference(_Tp)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_reference_v = __is_reference(_Tp);
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_lvalue_reference_v = __is_lvalue_reference(_Tp);
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_rvalue_reference_v = __is_rvalue_reference(_Tp);
#endif

#else // __has_builtin(__is_lvalue_reference) && etc...

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_lvalue_reference       : public false_type {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_lvalue_reference<_Tp&> : public true_type {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_rvalue_reference        : public false_type {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_rvalue_reference<_Tp&&> : public true_type {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_reference        : public false_type {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_reference<_Tp&>  : public true_type {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_reference<_Tp&&> : public true_type {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_reference_v = is_reference<_Tp>::value;

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_lvalue_reference_v = is_lvalue_reference<_Tp>::value;

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_rvalue_reference_v = is_rvalue_reference<_Tp>::value;
#endif

#endif // __has_builtin(__is_lvalue_reference) && etc...

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H
