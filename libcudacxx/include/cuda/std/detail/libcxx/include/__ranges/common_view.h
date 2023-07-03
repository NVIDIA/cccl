// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_COMMON_VIEW_H
#define _LIBCUDACXX___RANGES_COMMON_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/copyable.h"
#include "../__iterator/common_iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17
template<view _View>
  requires (!common_range<_View> && copyable<iterator_t<_View>>)
#else
template<class _View, enable_if_t<view<_View>, int> = 0
                    , enable_if_t<!common_range<_View>, int> = 0
                    , enable_if_t<copyable<iterator_t<_View>>, int> = 0>
#endif
class common_view : public view_interface<common_view<_View>> {
  _View __base_ = _View();

public:
#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  common_view() requires default_initializable<_View> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr common_view() noexcept(is_nothrow_default_constructible_v<_View2>)
    : view_interface<common_view<_View>>() {}
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit common_view(_View __v) noexcept(is_nothrow_move_constructible_v<_View>)
    : __base_(_CUDA_VSTD::move(__v)) { }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View2>)
  { return __base_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>)
  { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() {
    if constexpr (random_access_range<_View> && sized_range<_View>) {
      return _CUDA_VRANGES::begin(__base_);
    } else {
      return common_iterator<iterator_t<_View>, sentinel_t<_View>>(_CUDA_VRANGES::begin(__base_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() const {
    if constexpr (random_access_range<const _View> && sized_range<const _View>) {
      return _CUDA_VRANGES::begin(__base_);
    } else {
      return common_iterator<iterator_t<const _View>, sentinel_t<const _View>>(_CUDA_VRANGES::begin(__base_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() {
    if constexpr (random_access_range<_View> && sized_range<_View>) {
      return _CUDA_VRANGES::begin(__base_) + _CUDA_VRANGES::size(__base_);
    } else {
      return common_iterator<iterator_t<_View>, sentinel_t<_View>>(_CUDA_VRANGES::end(__base_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() const {
    if constexpr (random_access_range<const _View> && sized_range<const _View>) {
      return _CUDA_VRANGES::begin(__base_) + _CUDA_VRANGES::size(__base_);
    } else {
      return common_iterator<iterator_t<const _View>, sentinel_t<const _View>>(_CUDA_VRANGES::end(__base_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() {
    return _CUDA_VRANGES::size(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() const {
    return _CUDA_VRANGES::size(__base_);
  }
};

template<class _Range>
common_view(_Range&&) -> common_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template<class _View>
inline constexpr bool enable_borrowed_range<common_view<_View>> = enable_borrowed_range<_View>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__common)
  struct __fn : __range_adaptor_closure<__fn> {
    _LIBCUDACXX_TEMPLATE(class _Range)
      (requires common_range<_Range>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Range&& __range) const
      noexcept(noexcept(_CUDA_VIEWS::all(_CUDA_VSTD::forward<_Range>(__range))))
      -> decltype(      _CUDA_VIEWS::all(_CUDA_VSTD::forward<_Range>(__range)))
      { return          _CUDA_VIEWS::all(_CUDA_VSTD::forward<_Range>(__range)); }

    template<class _Range>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Range&& __range) const
      noexcept(noexcept(common_view{_CUDA_VSTD::forward<_Range>(__range)}))
      -> decltype(      common_view{_CUDA_VSTD::forward<_Range>(__range)})
      { return          common_view{_CUDA_VSTD::forward<_Range>(__range)}; }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto common = __common::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_COMMON_VIEW_H
