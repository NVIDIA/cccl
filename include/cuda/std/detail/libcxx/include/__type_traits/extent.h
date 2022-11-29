//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_EXTENT_H
#define _LIBCUDACXX___TYPE_TRAITS_EXTENT_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#include <cstddef>
#else
#include "../__type_traits/integral_constant.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__array_extent)

template<class _Tp, size_t _Dim = 0>
struct _LIBCUDACXX_TEMPLATE_VIS extent
    : integral_constant<size_t, __array_extent(_Tp, _Dim)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, unsigned _Ip = 0>
_LIBCUDACXX_INLINE_VAR constexpr size_t extent_v = __array_extent(_Tp, _Ip);
#endif

#else // __has_builtin(__array_extent)

template <class _Tp, unsigned _Ip = 0> struct _LIBCUDACXX_TEMPLATE_VIS extent
    : public integral_constant<size_t, 0> {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS extent<_Tp[], 0>
    : public integral_constant<size_t, 0> {};
template <class _Tp, unsigned _Ip> struct _LIBCUDACXX_TEMPLATE_VIS extent<_Tp[], _Ip>
    : public integral_constant<size_t, extent<_Tp, _Ip-1>::value> {};
template <class _Tp, size_t _Np> struct _LIBCUDACXX_TEMPLATE_VIS extent<_Tp[_Np], 0>
    : public integral_constant<size_t, _Np> {};
template <class _Tp, size_t _Np, unsigned _Ip> struct _LIBCUDACXX_TEMPLATE_VIS extent<_Tp[_Np], _Ip>
    : public integral_constant<size_t, extent<_Tp, _Ip-1>::value> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, unsigned _Ip = 0>
_LIBCUDACXX_INLINE_VAR constexpr size_t extent_v = extent<_Tp, _Ip>::value;
#endif

#endif // __has_builtin(__array_extent)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_EXTENT_H
