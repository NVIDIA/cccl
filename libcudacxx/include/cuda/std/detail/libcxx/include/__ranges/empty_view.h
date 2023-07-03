// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_EMPTY_VIEW_H
#define _LIBCUDACXX___RANGES_EMPTY_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/is_object.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

_LIBCUDACXX_TEMPLATE(class _Tp)
  (requires _LIBCUDACXX_TRAIT(is_object, _Tp))
class empty_view : public view_interface<empty_view<_Tp>> {
public:
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY static constexpr _Tp* begin() noexcept { return nullptr; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY static constexpr _Tp* end() noexcept { return nullptr; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY static constexpr _Tp* data() noexcept { return nullptr; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY static constexpr size_t size() noexcept { return 0; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY static constexpr bool empty() noexcept { return true; }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template<class _Tp>
inline constexpr bool enable_borrowed_range<empty_view<_Tp>> = true;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS

#if defined(_LIBCUDACXX_COMPILER_MSVC)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr empty_view<_Tp> empty{};
#else // ^^^ _LIBCUDACXX_COMPILER_MSVC ^^^/ vvv !_LIBCUDACXX_COMPILER_MSVC vvv
template <class _Tp>
_LIBCUDACXX_CPO_ACCESSIBILITY empty_view<_Tp> empty{};
#endif // !_LIBCUDACXX_COMPILER_MSVC

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14


#endif // _LIBCUDACXX___RANGES_EMPTY_VIEW_H
