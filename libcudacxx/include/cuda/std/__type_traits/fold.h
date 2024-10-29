//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_FOLD_H
#define _LIBCUDACXX___TYPE_TRAITS_FOLD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/negation.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017

// In C++17 and later, we can use fold expressions to implement __fold_and_v and __fold_or_v.
template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_and_v = (_Preds && ...);

template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_or_v = (_Preds || ...);

template <bool... _Preds>
using __fold_and = bool_constant<__fold_and_v<_Preds...>>;

template <bool... _Preds>
using __fold_or = bool_constant<__fold_or_v<_Preds...>>;

#elif _CCCL_STD_VER >= 2014

// In C++14, we can use a helper class to implement __fold_and_v and __fold_or_v.
template <bool... _Preds>
struct __fold_helper;

template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_and_v =
  _IsSame<__fold_helper<true, _Preds...>, __fold_helper<_Preds..., true>>::value;

template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_or_v =
  !_IsSame<__fold_helper<false, _Preds...>, __fold_helper<_Preds..., false>>::value;

template <bool... _Preds>
using __fold_and = bool_constant<__fold_and_v<_Preds...>>;

template <bool... _Preds>
using __fold_or = bool_constant<__fold_or_v<_Preds...>>;

#else // ^^^ _CCCL_STD_VER >= 2014 / _CCCL_STD_VER < 2014 vvv

// Otherwise, we cannot define __fold_and_v and __fold_or_v since C++11 does not
// have variable templates. Instead we provide just __fold_and and __fold_or type
// aliases.
template <bool... _Preds>
struct __fold_helper;

template <bool... _Preds>
using __fold_and = _IsSame<__fold_helper<true, _Preds...>, __fold_helper<_Preds..., true>>;

template <bool... _Preds>
using __fold_or = _Not<_IsSame<__fold_helper<false, _Preds...>, __fold_helper<_Preds..., false>>>;

#endif // _CCCL_STD_VER < 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_FOLD_H
