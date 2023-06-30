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

#include "../__concepts/derived_from.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

struct view_base { };

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17

template<class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
class view_interface;

#else

template<class _Derived, enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int> = 0>
class view_interface;

#endif // _LIBCUDACXX_STD_VER < 17

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

_LIBCUDACXX_TEMPLATE(class _Op, class _Yp)
  (requires is_convertible_v<_Op*, view_interface<_Yp>*>)
_LIBCUDACXX_INLINE_VISIBILITY
void __is_derived_from_view_interface(const _Op*, const view_interface<_Yp>*);

#if _LIBCUDACXX_STD_VER > 17

template <class _Tp>
inline constexpr bool enable_view = derived_from<_Tp, view_base> ||
  requires { _CUDA_VRANGES::__is_derived_from_view_interface((_Tp*)nullptr, (_Tp*)nullptr); };

#else

template <class _Tp, class = void>
inline constexpr bool enable_view = derived_from<_Tp, view_base>;

template <class _Tp>
inline constexpr bool enable_view<_Tp,
  void_t<decltype(_CUDA_VRANGES::__is_derived_from_view_interface((_Tp*)nullptr, (_Tp*)nullptr))>> = true;
#endif // _LIBCUDACXX_STD_VER < 17

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_ENABLE_VIEW_H
