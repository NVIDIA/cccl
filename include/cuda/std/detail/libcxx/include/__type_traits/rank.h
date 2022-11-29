//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_RANK_H
#define _LIBCUDACXX___TYPE_TRAITS_RANK_H

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

// TODO: Enable using the builtin __array_rank when https://llvm.org/PR57133 is resolved
#if __has_builtin(__array_rank) && 0

template <class _Tp>
struct rank : integral_constant<size_t, __array_rank(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr size_t rank_v = __array_rank(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS rank
    : public integral_constant<size_t, 0> {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS rank<_Tp[]>
    : public integral_constant<size_t, rank<_Tp>::value + 1> {};
template <class _Tp, size_t _Np> struct _LIBCUDACXX_TEMPLATE_VIS rank<_Tp[_Np]>
    : public integral_constant<size_t, rank<_Tp>::value + 1> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr size_t rank_v = rank<_Tp>::value;
#endif

#endif // __has_builtin(__array_rank)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_RANK_H
