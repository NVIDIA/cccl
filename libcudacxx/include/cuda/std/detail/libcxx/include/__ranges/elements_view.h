//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H
#define _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/three_way_comparable.h"
#endif
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/equality_comparable.h"
#include "../__fwd/get.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_like.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/maybe_const.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <class _View, size_t _Np, bool _Const>
class __elements_view_iterator;

template <class _View, size_t _Np, bool _Const>
class __elements_view_sentinel;

#if _LIBCUDACXX_STD_VER > 17
template <class _Tp, size_t _Np>
concept __has_tuple_element = __tuple_like<_Tp>::value && (_Np < tuple_size_v<_Tp>);
#else
template <class _Tp, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __has_tuple_element_,
  requires()(
    requires(__tuple_like<_Tp>::value),
    requires(_Np::value < tuple_size<_Tp>::value)
  ));

template <class _Tp, size_t _Np>
_LIBCUDACXX_CONCEPT __has_tuple_element = _LIBCUDACXX_FRAGMENT(__has_tuple_element_, _Tp, integral_constant<size_t, _Np>);
#endif // _LIBCUDACXX_STD_VER < 20

template <class _Tp, size_t _Np, class = void>
inline constexpr bool __returnable_element = is_reference_v<_Tp>;

template <class _Tp, size_t _Np>
inline constexpr bool __returnable_element<_Tp, _Np, enable_if_t<move_constructible<tuple_element_t<_Np, _Tp>>>> = true;

#if _LIBCUDACXX_STD_VER > 17
template <input_range _View, size_t _Np>
  requires view<_View> && __has_tuple_element<range_value_t<_View>, _Np> &&
           __has_tuple_element<remove_reference_t<range_reference_t<_View>>, _Np> &&
           __returnable_element<range_reference_t<_View>, _Np>
#else
template <class _View, size_t _Np,
          enable_if_t<input_range<_View>, int> = 0,
          enable_if_t<view<_View>, int> = 0,
          enable_if_t<__has_tuple_element<range_value_t<_View>, _Np>, int> = 0,
          enable_if_t<__has_tuple_element<remove_reference_t<range_reference_t<_View>>, _Np>, int> = 0,
          enable_if_t<__returnable_element<range_reference_t<_View>, _Np>, int> = 0>
#endif // _LIBCUDACXX_STD_VER < 20
class elements_view : public view_interface<elements_view<_View, _Np>> {
private:
  template <bool _Const>
  using __iterator = __elements_view_iterator<_View, _Np, _Const>;

  template <bool _Const>
  using __sentinel = __elements_view_sentinel<_View, _Np, _Const>;

  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();
public:
#if _LIBCUDACXX_STD_VER > 17
  elements_view() requires default_initializable<_View> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires default_initializable<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr elements_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<elements_view<_View, _Np>>() {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit elements_view(_View __base) noexcept(is_nothrow_move_constructible_v<_View>)
    : view_interface<elements_view<_View, _Np>>(), __base_(_CUDA_VSTD::move(__base)) {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _View base() const& {
    return __base_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _View base() && {
    return _CUDA_VSTD::move(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!__simple_view<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto begin() {
    return __iterator</*_Const=*/false>(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto begin() const {
    return __iterator</*_Const=*/true>(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!__simple_view<_View2>) _LIBCUDACXX_AND (!common_range<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() {
    return __sentinel</*_Const=*/false>{_CUDA_VRANGES::end(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!__simple_view<_View2>) _LIBCUDACXX_AND common_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() {
    return __iterator</*_Const=*/false>{_CUDA_VRANGES::end(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires range<const _View2> _LIBCUDACXX_AND (!common_range<const _View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() const {
    return __sentinel</*_Const=*/true>{_CUDA_VRANGES::end(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() const {
    return __iterator</*_Const=*/true>{_CUDA_VRANGES::end(__base_)};
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size() {
    return _CUDA_VRANGES::size(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size() const {
    return _CUDA_VRANGES::size(__base_);
  }
};

template <class, size_t, class = void>
struct __elements_view_iterator_category_base {};

template <class _Base, size_t _Np>
struct __elements_view_iterator_category_base<_Base, _Np, enable_if_t<forward_range<_Base>>> {
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  static constexpr auto __get_iterator_category() noexcept {
    using _Result = decltype(_CUDA_VSTD::get<_Np>(*_CUDA_VSTD::declval<iterator_t<_Base>>()));
    using _Cat    = typename iterator_traits<iterator_t<_Base>>::iterator_category;

    if constexpr (!is_lvalue_reference_v<_Result>) {
      return input_iterator_tag{};
    } else if constexpr (derived_from<_Cat, random_access_iterator_tag>) {
      return random_access_iterator_tag{};
    } else {
      return _Cat{};
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  using iterator_category = decltype(__get_iterator_category());
};

template <class _View, size_t _Np, bool _Const>
class __elements_view_iterator : public __elements_view_iterator_category_base<__maybe_const<_Const, _View>, _Np> {
  template <class, size_t, bool >
  friend class __elements_view_iterator;

  template <class, size_t, bool >
  friend class __elements_view_sentinel;

  using _Base = __maybe_const<_Const, _View>;
  template<bool _OtherConst>
  using _Base2 = __maybe_const<_OtherConst, _View>;

  iterator_t<_Base> __current_ = iterator_t<_Base>();

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  static constexpr decltype(auto) __get_element(const iterator_t<_Base>& __i) {
    if constexpr (is_reference_v<range_reference_t<_Base>>) {
      return _CUDA_VSTD::get<_Np>(*__i);
    } else {
      using _Element = remove_cv_t<tuple_element_t<_Np, range_reference_t<_Base>>>;
#if defined(_LIBCUDACXX_COMPILER_MSVC) // MSVC does not copy with the static_cast
      return _Element(_CUDA_VSTD::get<_Np>(*__i));
#else // ^^^ _LIBCUDACXX_COMPILER_MSVC ^^^ / vvv !_LIBCUDACXX_COMPILER_MSVC vvv
      return static_cast<_Element>(_CUDA_VSTD::get<_Np>(*__i));
#endif // !_LIBCUDACXX_COMPILER_MSVC
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  static constexpr auto __get_iterator_concept() noexcept {
    if constexpr (random_access_range<_Base>) {
      return random_access_iterator_tag{};
    } else if constexpr (bidirectional_range<_Base>) {
      return bidirectional_iterator_tag{};
    } else if constexpr (forward_range<_Base>) {
      return forward_iterator_tag{};
    } else {
      return input_iterator_tag{};
    }
    _LIBCUDACXX_UNREACHABLE();
  }

public:
  using iterator_concept = decltype(__get_iterator_concept());
  using value_type       = remove_cvref_t<tuple_element_t<_Np, range_value_t<_Base>>>;
  using difference_type  = range_difference_t<_Base>;

#if _LIBCUDACXX_STD_VER > 17
  __elements_view_iterator() requires default_initializable<iterator_t<_Base>> = default;
#else
    _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
      (requires default_initializable<iterator_t<_Base2<_Const2>>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __elements_view_iterator() noexcept(
      is_nothrow_default_constructible_v<iterator_t<_Base2<_Const2>>>) {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __elements_view_iterator(iterator_t<_Base> __current) noexcept(
    is_nothrow_move_constructible_v<iterator_t<_Base>>)
      : __current_(_CUDA_VSTD::move(__current)) {}

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires _Const2 _LIBCUDACXX_AND convertible_to<iterator_t<_View>, iterator_t<_Base>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator(__elements_view_iterator<_View, _Np, !_Const2> __i)
      : __current_(_CUDA_VSTD::move(__i.__current_)) {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  const iterator_t<_Base>& base() const& noexcept { return __current_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  iterator_t<_Base> base() && { return _CUDA_VSTD::move(__current_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  decltype(auto) operator*() const { return __get_element(__current_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator& operator++() {
    ++__current_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires (!forward_range<_Base2<_Const2>>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  void operator++(int) { ++__current_; }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires forward_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator operator++(int) {
    auto temp = *this;
    ++__current_;
    return temp;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires bidirectional_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator& operator--() {
    --__current_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires bidirectional_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator operator--(int) {
    auto temp = *this;
    --__current_;
    return temp;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator& operator+=(difference_type __n) {
    __current_ += __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_iterator& operator-=(difference_type __n) {
    __current_ -= __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  decltype(auto) operator[](difference_type __n) const {
    return __get_element(__current_ + __n);
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator==(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_Base2<_Const2>>>) {
    return __x.__current_ == __y.__current_;
  }
#if _LIBCUDACXX_STD_VER < 20
  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator!=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_Base2<_Const2>>>) {
    return __x.__current_ != __y.__current_;
  }
#endif

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator<(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2<_Const2>>) {
    return __x.__current_ < __y.__current_;
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator>(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2<_Const2>>) {
    return __y < __x;
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator<=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2<_Const2>>) {
    return !(__y < __x);
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator>=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2<_Const2>>) {
    return !(__x < __y);
  }

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr auto
  operator<=>(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }
#endif

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator+(const __elements_view_iterator& __x, difference_type __y)
  _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(requires random_access_range<_Base2<_Const2>>) {
    return __elements_view_iterator{__x} += __y;
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator+(difference_type __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(requires random_access_range<_Base2<_Const2>>) {
    return __y + __x;
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator-(const __elements_view_iterator& __x, difference_type __y)
  _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(requires random_access_range<_Base2<_Const2>>) {
    return __elements_view_iterator{__x} -= __y;
  }

  template<bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator-(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(difference_type)(requires sized_sentinel_for<iterator_t<_Base2<_Const2>>, iterator_t<_Base2<_Const2>>>) {
    return __x.__current_ - __y.__current_;
  }
};

template <class _View, size_t _Np, bool _Const>
class __elements_view_sentinel {
private:
  using _Base                                        = __maybe_const<_Const, _View>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS sentinel_t<_Base> __end_ = sentinel_t<_Base>();

  template <class, size_t, bool >
  friend class __elements_view_sentinel;

  template <bool _AnyConst>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  static constexpr decltype(auto) __get_current(const __elements_view_iterator<_View, _Np, _AnyConst>& __iter) {
    return (__iter.__current_);
  }

public:
  __elements_view_sentinel() = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __elements_view_sentinel(sentinel_t<_Base> __end)
    : __end_(_CUDA_VSTD::move(__end)) {}

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
    (requires _Const2 _LIBCUDACXX_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __elements_view_sentinel(__elements_view_sentinel<_View, _Np, !_Const2> __other)
    : __end_(_CUDA_VSTD::move(__other.__end_)) {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  sentinel_t<_Base> base() const { return __end_; }

  template<bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator==(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
    return __get_current(__x) == __y.__end_;
  }
#if _LIBCUDACXX_STD_VER < 20
  template<bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator==(const __elements_view_sentinel& __y, const __elements_view_iterator<_View, _Np, _OtherConst>& __x)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
    return __get_current(__x) == __y.__end_;
  }
  template<bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator!=(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
    return __get_current(__x) != __y.__end_;
  }
  template<bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator!=(const __elements_view_sentinel& __y, const __elements_view_iterator<_View, _Np, _OtherConst>& __x)
  _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
    return __get_current(__x) != __y.__end_;
  }
#endif

  template<bool _OtherConst>
  static constexpr bool __sized_sentinel = sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>;

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator-(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)
    (requires __sized_sentinel<_OtherConst>) {
    return __get_current(__x) - __y.__end_;
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto operator-(const __elements_view_sentinel& __x, const __elements_view_iterator<_View, _Np, _OtherConst>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)
    (requires __sized_sentinel<_OtherConst>) {
    return __x.__end_ - __get_current(__y);
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _Tp, size_t _Np>
inline constexpr bool enable_borrowed_range<elements_view<_Tp, _Np>> = enable_borrowed_range<_Tp>;

template <class _Tp>
using keys_view = elements_view<_Tp, 0>;
template <class _Tp>
using values_view = elements_view<_Tp, 1>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__elements)

template <size_t _Np>
struct __fn : __range_adaptor_closure<__fn<_Np>> {
  template <class _Range>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
      /**/ noexcept(noexcept(elements_view<all_t<_Range&&>, _Np>(_CUDA_VSTD::forward<_Range>(__range))))
      /*------*/ -> decltype(elements_view<all_t<_Range&&>, _Np>(_CUDA_VSTD::forward<_Range>(__range))) {
    /*-------------*/ return elements_view<all_t<_Range&&>, _Np>(_CUDA_VSTD::forward<_Range>(__range));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
#if defined(_LIBCUDACXX_COMPILER_MSVC)
  template <size_t _Np>
  inline constexpr auto elements = __elements::__fn<_Np>{};
#else // ^^^ _LIBCUDACXX_COMPILER_MSVC ^^^ / vvv !_LIBCUDACXX_COMPILER_MSVC vvv
  template <size_t _Np>
  _LIBCUDACXX_CPO_ACCESSIBILITY auto elements = __elements::__fn<_Np>{};
#endif // !_LIBCUDACXX_COMPILER_MSVC
  _LIBCUDACXX_CPO_ACCESSIBILITY auto keys     = elements<0>;
  _LIBCUDACXX_CPO_ACCESSIBILITY auto values   = elements<1>;
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H
