//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_OBJECT_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_OBJECT_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_class.h>
#include <__type_traits/is_scalar.h>
#include <__type_traits/is_union.h>
#else
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_scalar.h"
#include "../__type_traits/is_union.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_object)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_object : _BoolConstant<__is_object(_Tp)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_object_v = __is_object(_Tp);
#endif

#else // __has_builtin(__is_object)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_object
    : public integral_constant<bool, is_scalar<_Tp>::value ||
                                     is_array<_Tp>::value  ||
                                     is_union<_Tp>::value  ||
                                     is_class<_Tp>::value  > {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_object_v = is_object<_Tp>::value;
#endif

#endif // __has_builtin(__is_object)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_OBJECT_H
