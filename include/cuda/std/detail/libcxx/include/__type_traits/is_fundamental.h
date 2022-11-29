//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_null_pointer.h>
#include <__type_traits/is_void.h>
#else
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_null_pointer.h"
#include "../__type_traits/is_void.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_fundamental)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_fundamental : _BoolConstant<__is_fundamental(_Tp)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_fundamental_v = __is_fundamental(_Tp);
#endif

#else // __has_builtin(__is_fundamental)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_fundamental
    : public integral_constant<bool, is_void<_Tp>::value        ||
                                     __is_nullptr_t<_Tp>::value ||
                                     is_arithmetic<_Tp>::value> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_fundamental_v = is_fundamental<_Tp>::value;
#endif

#endif // __has_builtin(__is_fundamental)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H
