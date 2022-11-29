//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_enum.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_null_pointer.h>
#include <__type_traits/is_pointer.h>
#else
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_arithmetic.h"
#include "../__type_traits/is_enum.h"
#include "../__type_traits/is_member_pointer.h"
#include "../__type_traits/is_null_pointer.h"
#include "../__type_traits/is_pointer.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_scalar)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_scalar : _BoolConstant<__is_scalar(_Tp)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_scalar_v = __is_scalar(_Tp);
#endif

#else // __has_builtin(__is_scalar)

template <class _Tp> struct __is_block : false_type {};
#if defined(_LIBCUDACXX_HAS_EXTENSION_BLOCKS)
template <class _Rp, class ..._Args> struct __is_block<_Rp (^)(_Args...)> : true_type {};
#endif

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_scalar
    : public integral_constant<bool, is_arithmetic<_Tp>::value     ||
                                     is_member_pointer<_Tp>::value ||
                                     is_pointer<_Tp>::value        ||
                                     __is_nullptr_t<_Tp>::value    ||
                                     __is_block<_Tp>::value        ||
                                     is_enum<_Tp>::value           > {};

template <> struct _LIBCUDACXX_TEMPLATE_VIS is_scalar<nullptr_t> : public true_type {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_scalar_v = is_scalar<_Tp>::value;
#endif

#endif // __has_builtin(__is_scalar)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
