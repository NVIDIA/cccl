//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H

#ifndef __cuda_std__
#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_same.h>
#else
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_same.h"
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__is_referenceable)

template <class _Tp>
struct __libcpp_is_referenceable : integral_constant<bool, __is_referenceable(_Tp)> {};

#else
struct __libcpp_is_referenceable_impl {
  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static _Tp& __test(int);
  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static false_type __test(...);
};

template <class _Tp>
struct __libcpp_is_referenceable
    : integral_constant<bool, _IsNotSame<decltype(__libcpp_is_referenceable_impl::__test<_Tp>(0)), false_type>::value> {
};
#endif // __has_builtin(__is_referenceable)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
