// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_FILTER_VIEW_H
#define _LIBCUDACXX___RANGES_FILTER_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/ranges_find_if.h"
#include "../__concepts/constructible.h"
#include "../__concepts/copyable.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/equality_comparable.h"
#include "../__debug"
#include "../__functional/bind_back.h"
#include "../__functional/invoke.h"
#include "../__functional/reference_wrapper.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/non_propagating_cache.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_object.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template<class _View, class = void>
struct __filter_iterator_category { };

template<class _View>
struct __filter_iterator_category<_View, enable_if_t<forward_range<_View>>> {
  using _Cat = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using iterator_category =
    _If<derived_from<_Cat, bidirectional_iterator_tag>, bidirectional_iterator_tag,
    _If<derived_from<_Cat, forward_iterator_tag>,       forward_iterator_tag,
    /* else */                                          _Cat
  >>;
};

#if _LIBCUDACXX_STD_VER > 17
template<input_range _View, indirect_unary_predicate<iterator_t<_View>> _Pred>
  requires view<_View> && is_object_v<_Pred>
#else
template<class _View, class _Pred,
          class = enable_if_t<view<_View>>,
          class = enable_if_t<input_range<_View>>,
          class = enable_if_t<is_object_v<_Pred>>,
          class = enable_if_t<indirect_unary_predicate<_Pred, iterator_t<_View>>>>
#endif // _LIBCUDACXX_STD_VER < 20
class filter_view : public view_interface<filter_view<_View, _Pred>> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __copyable_box<_Pred> __pred_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();

  // We cache the result of begin() to allow providing an amortized O(1) begin() whenever
  // the underlying range is at least a forward_range.
  static constexpr bool _UseCache = forward_range<_View>;
  using _Cache = _If<_UseCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();

public:
  class __iterator : public __filter_iterator_category<_View> {
  public:
    _LIBCUDACXX_NO_UNIQUE_ADDRESS iterator_t<_View> __current_ = iterator_t<_View>();
    _LIBCUDACXX_NO_UNIQUE_ADDRESS filter_view* __parent_ = nullptr;

    using iterator_concept =
      _If<bidirectional_range<_View>, bidirectional_iterator_tag,
      _If<forward_range<_View>,       forward_iterator_tag,
      /* else */                      input_iterator_tag
    >>;
    // using iterator_category = inherited;
    using value_type = range_value_t<_View>;
    using difference_type = range_difference_t<_View>;

// It seems nvcc has a bug where the noexcept specification is incorrectly set
#if 0 // _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI
    __iterator() requires default_initializable<iterator_t<_View>> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires default_initializable<iterator_t<_View2>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator() noexcept(is_nothrow_default_constructible_v<iterator_t<_View2>>) {}
#endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator(filter_view& __parent, iterator_t<_View> __current)
      : __current_(_CUDA_VSTD::move(__current)), __parent_(_CUDA_VSTD::addressof(__parent))
    { }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iterator_t<_View> const& base() const& noexcept { return __current_; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iterator_t<_View> base() && { return _CUDA_VSTD::move(__current_); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr range_reference_t<_View> operator*() const { return *__current_; }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires __has_arrow<iterator_t<_View2>> && copyable<iterator_t<_View2>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iterator_t<_View> operator->() const {
      return __current_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator++() {
      __current_ = _CUDA_VRANGES::find_if(_CUDA_VSTD::move(++__current_), _CUDA_VRANGES::end(__parent_->__base_),
                                    _CUDA_VSTD::ref(*__parent_->__pred_));
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires (!forward_range<_View2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void operator++(int) { ++*this; }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires forward_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator operator++(int) {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires bidirectional_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator--() {
      do {
        --__current_;
      } while (!_CUDA_VSTD::invoke(*__parent_->__pred_, *__current_));
      return *this;
    }
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires bidirectional_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator operator--(int) {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    template<class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    auto operator==(__iterator const& __x, __iterator const& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_View2>>)
    {
      return __x.__current_ == __y.__current_;
    }
#if _LIBCUDACXX_STD_VER < 20
    template<class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    auto operator!=(__iterator const& __x, __iterator const& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_View2>>)
    {
      return __x.__current_ != __y.__current_;
    }
#endif

    // MSVC falls over its feet if this is not a template
    template<class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    range_rvalue_reference_t<_View2> iter_move(__iterator const& __it)
      noexcept(noexcept(_CUDA_VRANGES::iter_move(__it.__current_)))
    {
      return _CUDA_VRANGES::iter_move(__it.__current_);
    }


    template<class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    auto iter_swap(__iterator const& __x, __iterator const& __y)
      noexcept(__noexcept_swappable<iterator_t<_View2>>)
      _LIBCUDACXX_TRAILING_REQUIRES(void)(requires indirectly_swappable<iterator_t<_View2>>)
    {
      return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
    }
  };

  class __sentinel {
  public:
    sentinel_t<_View> __end_ = sentinel_t<_View>();

    _LIBCUDACXX_HIDE_FROM_ABI
    __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __sentinel(filter_view& __parent)
      : __end_(_CUDA_VRANGES::end(__parent.__base_))
    { }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr sentinel_t<_View> base() const { return __end_; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    bool operator==(__iterator const& __x, __sentinel const& __y) {
      return __x.__current_ == __y.__end_;
    }
#if _LIBCUDACXX_STD_VER < 20
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    bool operator==(__sentinel const& __x, __iterator const& __y) {
      return __y.__current_ == __x.__end_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    bool operator!=(__iterator const& __x, __sentinel const& __y) {
      return __x.__current_ != __y.__end_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr
    bool operator!=(__sentinel const& __x, __iterator const& __y) {
      return __y.__current_ != __x.__end_;
    }
#endif
  };

#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  filter_view() requires default_initializable<_View> && default_initializable<_Pred> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires default_initializable<_View2> && default_initializable<_Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr filter_view() noexcept(is_nothrow_default_constructible_v<_View2>
                                && is_nothrow_default_constructible_v<_Pred>)
    : view_interface<filter_view<_View, _Pred>>() {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr filter_view(_View __base, _Pred __pred)
    : view_interface<filter_view<_View, _Pred>>()
    , __pred_(in_place, _CUDA_VSTD::move(__pred))
    , __base_(_CUDA_VSTD::move(__base))
  { }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& { return __base_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Pred const& pred() const { return *__pred_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __iterator begin() {
    _LIBCUDACXX_ASSERT(__pred_.__has_value(), "Trying to call begin() on a filter_view that does not have a valid predicate.");
    if constexpr (_UseCache) {
      if (!__cached_begin_.__has_value()) {
        __cached_begin_.__emplace(_CUDA_VRANGES::find_if(__base_, _CUDA_VSTD::ref(*__pred_)));
      }
      return {*this, *__cached_begin_};
    } else {
      return {*this, _CUDA_VRANGES::find_if(__base_, _CUDA_VSTD::ref(*__pred_))};
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() {
    if constexpr (common_range<_View>) {
      return __iterator{*this, _CUDA_VRANGES::end(__base_)};
    } else {
      return __sentinel{*this};
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};

template<class _Range, class _Pred>
filter_view(_Range&&, _Pred) -> filter_view<_CUDA_VIEWS::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__filter)
  struct __fn {
    template<class _Range, class _Pred>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
      noexcept(noexcept(filter_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))))
      -> decltype(      filter_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred)))
      { return          filter_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred)); }

    _LIBCUDACXX_TEMPLATE(class _Pred)
      (requires constructible_from<decay_t<_Pred>, _Pred>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Pred&& __pred) const
      noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>)
    { return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pred>(__pred))); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto filter = __filter::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_FILTER_VIEW_H
