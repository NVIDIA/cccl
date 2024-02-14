// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
#define _LIBCUDACXX___RANGES_VIEW_INTERFACE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__assert"
#include "../__concepts/derived_from.h"
#include "../__concepts/same_as.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/prev.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/empty.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_cv.h"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#  if _CCCL_STD_VER >= 2020
template <class _Tp>
concept __can_empty = requires(_Tp& __t) { _CUDA_VRANGES::empty(__t); };
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(__can_empty_, requires(_Tp& __t)(typename(decltype(_CUDA_VRANGES::empty(__t)))));

template <class _Tp>
_LIBCUDACXX_CONCEPT __can_empty = _LIBCUDACXX_FRAGMENT(__can_empty_, _Tp);
#  endif //  _CCCL_STD_VER <= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Derived, enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int>>
#  endif //  _CCCL_STD_VER <= 2017
class view_interface
{
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Derived& __derived() noexcept
  {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived&>(*this);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Derived const& __derived() const noexcept
  {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived const&>(*this);
  }

public:
  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(forward_range<_D2>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty()
  {
    return _CUDA_VRANGES::begin(__derived()) == _CUDA_VRANGES::end(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(forward_range<const _D2>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty() const
  {
    return _CUDA_VRANGES::begin(__derived()) == _CUDA_VRANGES::end(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(__can_empty<_D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit operator bool()
  {
    return !_CUDA_VRANGES::empty(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(__can_empty<const _D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit operator bool() const
  {
    return !_CUDA_VRANGES::empty(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(contiguous_iterator<iterator_t<_D2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto data()
  {
    return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(range<const _D2> _LIBCUDACXX_AND contiguous_iterator<iterator_t<const _D2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto data() const
  {
    return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(forward_range<_D2> _LIBCUDACXX_AND sized_sentinel_for<sentinel_t<_D2>, iterator_t<_D2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size()
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__derived()) - _CUDA_VRANGES::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(
    forward_range<const _D2> _LIBCUDACXX_AND sized_sentinel_for<sentinel_t<const _D2>, iterator_t<const _D2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size() const
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__derived()) - _CUDA_VRANGES::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(forward_range<_D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto) front()
  {
    _LIBCUDACXX_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *_CUDA_VRANGES::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(forward_range<const _D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto) front() const
  {
    _LIBCUDACXX_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *_CUDA_VRANGES::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(bidirectional_range<_D2> _LIBCUDACXX_AND common_range<_D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto) back()
  {
    _LIBCUDACXX_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *_CUDA_VRANGES::prev(_CUDA_VRANGES::end(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
  _LIBCUDACXX_REQUIRES(bidirectional_range<const _D2> _LIBCUDACXX_AND common_range<const _D2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto) back() const
  {
    _LIBCUDACXX_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *_CUDA_VRANGES::prev(_CUDA_VRANGES::end(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _RARange = _Derived)
  _LIBCUDACXX_REQUIRES(random_access_range<_RARange>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto)
  operator[](range_difference_t<_RARange> __index)
  {
    return _CUDA_VRANGES::begin(__derived())[__index];
  }

  _LIBCUDACXX_TEMPLATE(class _RARange = const _Derived)
  _LIBCUDACXX_REQUIRES(random_access_range<_RARange>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr decltype(auto)
  operator[](range_difference_t<_RARange> __index) const
  {
    return _CUDA_VRANGES::begin(__derived())[__index];
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
