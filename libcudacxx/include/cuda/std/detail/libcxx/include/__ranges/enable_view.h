// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_ENABLE_VIEW_H
#define _LIBCUDACXX___RANGES_ENABLE_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/derived_from.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/void_t.h"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017

struct view_base { };

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _CCCL_STD_VER >= 2020

template<class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
class view_interface;

#else // ^^^ _CCCL_STD_VER >= 2020 ^^^ / vvv _CCCL_STD_VER <= 2017 vvv

template<class _Derived, enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int> = 0>
class view_interface;

#endif // _CCCL_STD_VER <= 2017

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

_LIBCUDACXX_TEMPLATE(class _Op, class _Yp)
  _LIBCUDACXX_REQUIRES(is_convertible_v<_Op*, view_interface<_Yp>*>)
_LIBCUDACXX_INLINE_VISIBILITY
void __is_derived_from_view_interface(const _Op*, const view_interface<_Yp>*);

#if _CCCL_STD_VER >= 2020

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_view = derived_from<_Tp, view_base> ||
  requires { _CUDA_VRANGES::__is_derived_from_view_interface((_Tp*)nullptr, (_Tp*)nullptr); };

#else // ^^^ _CCCL_STD_VER >= 2020 ^^^ / vvv _CCCL_STD_VER <= 2017 vvv

template <class _Tp, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_view = derived_from<_Tp, view_base>;

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_view<_Tp,
  void_t<decltype(_CUDA_VRANGES::__is_derived_from_view_interface((_Tp*)nullptr, (_Tp*)nullptr))>> = true;
#endif // _CCCL_STD_VER <= 2017

#endif // _CCCL_STD_VER >= 2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_ENABLE_VIEW_H
