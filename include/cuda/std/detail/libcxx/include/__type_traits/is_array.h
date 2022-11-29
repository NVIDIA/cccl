//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H

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

// TODO: Clang incorrectly reports that __is_array is true for T[0].
//       Re-enable the branch once https://llvm.org/PR54705 is fixed.
#if __has_builtin(__is_array) && 0

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_array : _BoolConstant<__is_array(_Tp)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_array_v = __is_array(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_array
    : public false_type {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_array<_Tp[]>
    : public true_type {};
template <class _Tp, size_t _Np> struct _LIBCUDACXX_TEMPLATE_VIS is_array<_Tp[_Np]>
    : public true_type {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_array_v = is_array<_Tp>::value;
#endif

#endif // __has_builtin(__is_array)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
