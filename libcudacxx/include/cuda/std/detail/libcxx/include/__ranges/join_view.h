// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_JOIN_VIEW_H
#define _LIBCUDACXX___RANGES_JOIN_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/copyable.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/equality_comparable.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/defaultable_box.h"
#include "../__ranges/non_propagating_cache.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/maybe_const.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17
template<class>
struct __join_view_iterator_category {};

template<class _View>
  requires is_reference_v<range_reference_t<_View>> &&
           forward_range<_View> &&
           forward_range<range_reference_t<_View>>
struct __join_view_iterator_category<_View> {
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
template<class, class = void, class = void, class = void>
struct __join_view_iterator_category {};

template<class _View>
struct __join_view_iterator_category<_View, enable_if_t<is_reference_v<range_reference_t<_View>>>
                                          , enable_if_t<forward_range<_View>>
                                          , enable_if_t<forward_range<range_reference_t<_View>>>> {
#endif // _LIBCUDACXX_STD_VER < 20
  using _OuterC = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using _InnerC = typename iterator_traits<iterator_t<range_reference_t<_View>>>::iterator_category;

  using iterator_category = _If<
    derived_from<_OuterC, bidirectional_iterator_tag> && derived_from<_InnerC, bidirectional_iterator_tag> &&
      common_range<range_reference_t<_View>>,
    bidirectional_iterator_tag,
    _If<
      derived_from<_OuterC, forward_iterator_tag> && derived_from<_InnerC, forward_iterator_tag>,
      forward_iterator_tag,
      input_iterator_tag
    >
  >;
};

// We only need to cache the variable if the view has a non reference reference type
template <class _View, class = void>
struct __join_view_base {
private:
  struct __cache_wrapper {
    remove_cv_t<range_reference_t<_View>> __value_;

    template <class _Iter, enable_if_t<input_iterator<_Iter>, int> = 0>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __cache_wrapper(in_place_t, const _Iter& __iter) noexcept(
        noexcept(static_cast<decltype(__value_)>(*__iter)))
        : __value_(*__iter) {}
  };

protected:
    _LIBCUDACXX_NO_UNIQUE_ADDRESS __non_propagating_cache<__cache_wrapper> __cache_{};
};

template <class _View>
class __join_view_base<_View, enable_if_t<is_reference_v<range_reference_t<_View>>>> {};

#if _LIBCUDACXX_STD_VER > 17
template<input_range _View>
  requires view<_View> && input_range<range_reference_t<_View>>
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
template<class _View, enable_if_t<view<_View>, int> = 0
                    , enable_if_t<input_range<_View>, int> = 0
                    , enable_if_t<input_range<range_reference_t<_View>>, int> = 0>
#endif // _LIBCUDACXX_STD_VER < 20
class join_view : public __join_view_base<_View>, public view_interface<join_view<_View>> {
private:
  using _InnerRange = range_reference_t<_View>;

  static constexpr bool _UseCache = !is_reference_v<_InnerRange>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
  template<bool _Const>
  struct __iterator : public __join_view_iterator_category<__maybe_const<_Const, _View>> {

  template<bool> friend struct __iterator;

  private:
    using _Parent = __maybe_const<_Const, join_view>;
    using _Base = __maybe_const<_Const, _View>;
    using _Outer = iterator_t<_Base>;
    using _Inner = iterator_t<range_reference_t<_Base>>;

    static constexpr bool __ref_is_glvalue = is_reference_v<range_reference_t<_Base>>;

public:
    _Outer __outer_ = _Outer();

private:
    __defaultable_box<_Inner> __inner_{};
    _Parent *__parent_ = nullptr;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto&& __update_inner() {
      if constexpr (__ref_is_glvalue) {
          return *__outer_;
      } else {
          return __parent_->__cache_.__emplace(in_place, __outer_).__value_;
      }
      _LIBCUDACXX_UNREACHABLE();
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void __satisfy() {
      for (; __outer_ != _CUDA_VRANGES::end(__parent_->__base_); ++__outer_) {
        auto&& __inner = __update_inner();
        __inner_ = _CUDA_VRANGES::begin(__inner);
        if (*__inner_ != _CUDA_VRANGES::end(__inner))
          return;
      }

      if constexpr (__ref_is_glvalue)
        __inner_.__reset();
    }

  public:
    using iterator_concept = _If<
      __ref_is_glvalue && bidirectional_range<_Base> && bidirectional_range<range_reference_t<_Base>> &&
          common_range<range_reference_t<_Base>>,
      bidirectional_iterator_tag,
      _If<
        __ref_is_glvalue && forward_range<_Base> && forward_range<range_reference_t<_Base>>,
        forward_iterator_tag,
        input_iterator_tag
      >
    >;

    using value_type = range_value_t<range_reference_t<_Base>>;

    using difference_type = common_type_t<
      range_difference_t<_Base>, range_difference_t<range_reference_t<_Base>>>;

#if _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI
    __iterator() requires default_initializable<_Outer> = default;
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
    _LIBCUDACXX_TEMPLATE(class _Outer2 = _Outer)
      (requires default_initializable<_Outer2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator() noexcept(is_nothrow_default_constructible_v<_Outer2>) {}
#endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator(_Parent& __parent, _Outer __outer)
      : __outer_(_CUDA_VSTD::move(__outer))
      , __parent_(_CUDA_VSTD::addressof(__parent)) {
      __satisfy();
    }

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      (requires _OtherConst _LIBCUDACXX_AND
                convertible_to<iterator_t<_View>, _Outer> _LIBCUDACXX_AND
                convertible_to<iterator_t<_InnerRange>, _Inner>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator(__iterator<!_OtherConst> __i)
      : __outer_(_CUDA_VSTD::move(__i.__outer_))
      , __inner_(_CUDA_VSTD::move(__i.__inner_))
      , __parent_(__i.__parent_) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator*() const {
      return **__inner_;
    }

    _LIBCUDACXX_TEMPLATE(class _Inner2 = _Inner)
      (requires __has_arrow<_Inner2> _LIBCUDACXX_AND copyable<_Inner2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _Inner2 operator->() const {
      return *__inner_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator++() {
      if constexpr (__ref_is_glvalue) {
        if (++*__inner_ == _CUDA_VRANGES::end(*__outer_)) {
          ++__outer_;
          __satisfy();
        }
      } else {
        if (++*__inner_ == _CUDA_VRANGES::end((*__parent_->__cache_).__value_)) {
          ++__outer_;
          __satisfy();
        }
      }
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator++(int) {
      if constexpr(__ref_is_glvalue &&
                   forward_range<_Base> &&
                   forward_range<range_reference_t<_Base>>) {
        auto __tmp = *this;
        ++*this;
        return __tmp;
      } else {
        ++*this;
      }
    }

    _LIBCUDACXX_TEMPLATE(bool __ref_is_glvalue2 = __ref_is_glvalue)
      (requires __ref_is_glvalue2 _LIBCUDACXX_AND
                bidirectional_range<_Base> _LIBCUDACXX_AND
                bidirectional_range<range_reference_t<_Base>> _LIBCUDACXX_AND
                common_range<range_reference_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator--() {
      if (__outer_ == _CUDA_VRANGES::end(__parent_->__base_))
        __inner_ = _CUDA_VRANGES::end(*--__outer_);

      // Skip empty inner ranges when going backwards.
      while (*__inner_ == _CUDA_VRANGES::begin(*__outer_)) {
        __inner_ = _CUDA_VRANGES::end(*--__outer_);
      }

      --*__inner_;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(bool __ref_is_glvalue2 = __ref_is_glvalue)
      (requires __ref_is_glvalue2 _LIBCUDACXX_AND
                bidirectional_range<_Base> _LIBCUDACXX_AND
                bidirectional_range<range_reference_t<_Base>> _LIBCUDACXX_AND
                common_range<range_reference_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator operator--(int) {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    template<bool __ref_is_glvalue2 = __ref_is_glvalue>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __iterator& __x, const __iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires __ref_is_glvalue2 &&
      equality_comparable<iterator_t<_Base>> && equality_comparable<iterator_t<range_reference_t<_Base>>>) {
      return __x.__outer_ == __y.__outer_ && __x.__inner_ == __y.__inner_;
    }
#if _LIBCUDACXX_STD_VER < 20
    template<bool __ref_is_glvalue2 = __ref_is_glvalue>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __iterator& __x, const __iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires __ref_is_glvalue2 &&
      equality_comparable<iterator_t<_Base>> && equality_comparable<iterator_t<range_reference_t<_Base>>>) {
      return __x.__outer_ != __y.__outer_ || __x.__inner_ != __y.__inner_;
    }
#endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr decltype(auto) iter_move(const __iterator& __i)
      noexcept(noexcept(_CUDA_VRANGES::iter_move(*__i.__inner_))) {
      return _CUDA_VRANGES::iter_move(*__i.__inner_);
    }

    template<class _Inner2 = _Inner>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto iter_swap(const __iterator& __x, const __iterator& __y)
      noexcept(__noexcept_swappable<_Inner2>)
      _LIBCUDACXX_TRAILING_REQUIRES(void)(requires indirectly_swappable<_Inner2>) {
      return _CUDA_VRANGES::iter_swap(*__x.__inner_, *__y.__inner_);
    }
  };

  template<bool _Const>
  struct __sentinel {

    template<bool>
    friend struct __sentinel;

  private:
    using _Parent = __maybe_const<_Const, join_view>;
    using _Base = __maybe_const<_Const, _View>;
    sentinel_t<_Base> __end_ = sentinel_t<_Base>();

  public:
    _LIBCUDACXX_HIDE_FROM_ABI
    __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __sentinel(_Parent& __parent)
      : __end_(_CUDA_VRANGES::end(__parent.__base_)) {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      (requires _OtherConst _LIBCUDACXX_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __sentinel(__sentinel<!_OtherConst> __s)
      : __end_(_CUDA_VSTD::move(__s.__end_)) {}

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
      (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__outer_ == __y.__end_;
    }
#if _LIBCUDACXX_STD_VER < 20
    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
      (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __y.__outer_ == __x.__end_;
    }
    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
      (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__outer_ != __y.__end_;
    }
    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
      (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __y.__outer_ != __x.__end_;
    }
#endif // _LIBCUDACXX_STD_VER < 20
  };

#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  join_view() requires default_initializable<_View> = default;
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr join_view() noexcept(is_nothrow_default_constructible_v<_View2>)
    : view_interface<join_view<_View>>() {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit join_view(_View __base)
    : view_interface<join_view<_View>>()
    , __base_(_CUDA_VSTD::move(__base)) {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& { return __base_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() {
    constexpr bool __use_const = __simple_view<_View> &&
                                  is_reference_v<range_reference_t<_View>>;
    return __iterator<__use_const>{*this, _CUDA_VRANGES::begin(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires input_range<const _View2> _LIBCUDACXX_AND is_reference_v<range_reference_t<const _View2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() const {
    return __iterator<true>{*this, _CUDA_VRANGES::begin(__base_)};
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() {
    if constexpr (forward_range<_View> &&
                  is_reference_v<_InnerRange> &&
                  forward_range<_InnerRange> &&
                  common_range<_View> &&
                  common_range<_InnerRange>) {
      return __iterator<__simple_view<_View>>{*this, _CUDA_VRANGES::end(__base_)};
    } else {
      return __sentinel<__simple_view<_View>>{*this};
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires input_range<const _View2> _LIBCUDACXX_AND is_reference_v<range_reference_t<const _View2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() const {
    using _ConstInnerRange = range_reference_t<const _View>;
    if constexpr (forward_range<const _View> &&
                  is_reference_v<_ConstInnerRange> &&
                  forward_range<_ConstInnerRange> &&
                  common_range<const _View> &&
                  common_range<_ConstInnerRange>) {
      return __iterator<true>{*this, _CUDA_VRANGES::end(__base_)};
    } else {
      return __sentinel<true>{*this};
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};

template<class _Range>
explicit join_view(_Range&&) -> join_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__join_view)
struct __fn : __range_adaptor_closure<__fn> {
  template<class _Range>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(join_view<all_t<_Range&&>>(_CUDA_VSTD::forward<_Range>(__range))))
    -> decltype(      join_view<all_t<_Range&&>>(_CUDA_VSTD::forward<_Range>(__range)))
    { return          join_view<all_t<_Range&&>>(_CUDA_VSTD::forward<_Range>(__range)); }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto join = __join_view::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_JOIN_VIEW_H
