//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_valid_expansion.h"
#include "../__type_traits/void_t.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_COMPILER_MSVC)
template<class _Tp, class = void>
struct __is_primary_template : false_type {};

template<class _Tp>
struct __is_primary_template<_Tp, void_t<typename _Tp::__primary_template>>
  : public is_same<_Tp, typename _Tp::__primary_template> {};

#else // ^^^ _LIBCUDACXX_COMPILER_MSVC ^^^ / vvv !_LIBCUDACXX_COMPILER_MSVC vvv

template <class _Tp>
using __test_for_primary_template = __enable_if_t<
    _IsSame<_Tp, typename _Tp::__primary_template>::value
  >;
template <class _Tp>
using __is_primary_template = _IsValidExpansion<
    __test_for_primary_template, _Tp
  >;
#endif // !_LIBCUDACXX_COMPILER_MSVC

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
