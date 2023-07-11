// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__concepts/assignable.h"
#include "../__concepts/common_with.h"
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/same_as.h"
#include "../__iterator/concepts.h"
#include "../__iterator/default_sentinel.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iter_swap.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/readable_traits.h"
#include "../__memory/pointer_traits.h"
#include "../__type_traits/add_pointer.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_assignable.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/void_t.h"
#include "../__utility/move.h"

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../compare"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 4848)
#endif // _LIBCUDACXX_COMPILER_MSVC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

template<class, class = void>
struct __counted_iterator_concept {};

template<class _Iter>
struct __counted_iterator_concept<_Iter, void_t<typename _Iter::iterator_concept>> {
  using iterator_concept = typename _Iter::iterator_concept;
};

template<class, class = void>
struct __counted_iterator_category {};

template<class _Iter>
struct __counted_iterator_category<_Iter, void_t<typename _Iter::iterator_category>> {
  using iterator_category = typename _Iter::iterator_category;
};

template<class, class = void>
struct __counted_iterator_value_type {};

template<class _Iter>
struct __counted_iterator_value_type<_Iter, enable_if_t<indirectly_readable<_Iter>>> {
  using value_type = iter_value_t<_Iter>;
};

#if _LIBCUDACXX_STD_VER > 17
template<input_or_output_iterator _Iter>
#else
template<class _Iter, enable_if_t<input_or_output_iterator<_Iter>, int> = 0>
#endif
class counted_iterator
  : public __counted_iterator_concept<_Iter>
  , public __counted_iterator_category<_Iter>
  , public __counted_iterator_value_type<_Iter>
{
public:
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Iter __current_ = _Iter();
  iter_difference_t<_Iter> __count_ = 0;

  using iterator_type = _Iter;
  using difference_type = iter_difference_t<_Iter>;

#if _LIBCUDACXX_STD_VER > 17
  counted_iterator() requires default_initializable<_Iter> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires default_initializable<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator() noexcept(is_nothrow_default_constructible_v<_I2>) {}
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator(_Iter __iter, iter_difference_t<_Iter> __n)
    noexcept(is_nothrow_move_constructible_v<_Iter>)
   : __current_(_CUDA_VSTD::move(__iter)), __count_(__n) {
    _LIBCUDACXX_ASSERT(__n >= 0, "__n must not be negative.");
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires convertible_to<const _I2&, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator(const counted_iterator<_I2>& __other)
    noexcept(is_nothrow_constructible_v<_Iter, const _I2&>)
   : __current_(__other.__current_), __count_(__other.__count_) {}

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires assignable_from<_Iter&, const _I2&>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator& operator=(const counted_iterator<_I2>& __other)
    noexcept(is_nothrow_assignable_v<_Iter&, const _I2&>) {
    __current_ = __other.__current_;
    __count_ = __other.__count_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr const _Iter& base() const& noexcept { return __current_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Iter base() && { return _CUDA_VSTD::move(__current_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr iter_difference_t<_Iter> count() const noexcept { return __count_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) operator*() {
    _LIBCUDACXX_ASSERT(__count_ > 0, "Iterator is equal to or past end.");
    return *__current_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires __dereferenceable<const _I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) operator*() const
  {
    _LIBCUDACXX_ASSERT(__count_ > 0, "Iterator is equal to or past end.");
    return *__current_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires contiguous_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator->() const noexcept
  {
    return _CUDA_VSTD::to_address(__current_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator& operator++() {
    _LIBCUDACXX_ASSERT(__count_ > 0, "Iterator already at or past end.");
    ++__current_;
    --__count_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires (!forward_iterator<_I2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  decltype(auto) operator++(int) {
    _LIBCUDACXX_ASSERT(__count_ > 0, "Iterator already at or past end.");
    --__count_;
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
    try { return __current_++; }
    catch(...) { ++__count_; throw; }
#else
    return __current_++;
#endif // _LIBCUDACXX_NO_EXCEPTIONS
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires forward_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator operator++(int)
  {
    _LIBCUDACXX_ASSERT(__count_ > 0, "Iterator already at or past end.");
    counted_iterator __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires bidirectional_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator& operator--()
  {
    --__current_;
    ++__count_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires bidirectional_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator operator--(int)
  {
    counted_iterator __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator operator+(iter_difference_t<_I2> __n) const
  {
    return counted_iterator(__current_ + __n, __count_ - __n);
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr counted_iterator operator+(
    iter_difference_t<_I2> __n, const counted_iterator& __x)
  {
    return __x + __n;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator& operator+=(iter_difference_t<_I2> __n)
  {
    _LIBCUDACXX_ASSERT(__n <= __count_, "Cannot advance iterator past end.");
    __current_ += __n;
    __count_ -= __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator operator-(iter_difference_t<_I2> __n) const
  {
    return counted_iterator(__current_ - __n, __count_ + __n);
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr iter_difference_t<_I2> operator-(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __rhs.__count_ - __lhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr iter_difference_t<_I2> operator-(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __rhs.__count_ - __lhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr iter_difference_t<_Iter> operator-(
    const counted_iterator& __lhs, default_sentinel_t)
  {
    return -__lhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr iter_difference_t<_Iter> operator-(
    default_sentinel_t, const counted_iterator& __rhs)
  {
    return __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr counted_iterator& operator-=(iter_difference_t<_I2> __n)
  {
    _LIBCUDACXX_ASSERT(-__n <= __count_, "Attempt to subtract too large of a size: "
                                     "counted_iterator would be decremented before the "
                                     "first element of its range.");
    __current_ -= __n;
    __count_ += __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
    (requires random_access_iterator<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) operator[](iter_difference_t<_I2> __n) const
  {
    _LIBCUDACXX_ASSERT(__n < __count_, "Subscript argument must be less than size.");
    return __current_[__n];
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ == __rhs.__count_;
  }

#if _LIBCUDACXX_STD_VER < 20
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ == __rhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ != __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ != __rhs.__count_;
  }
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(
    const counted_iterator& __lhs, default_sentinel_t)
  {
    return __lhs.__count_ == 0;
  }

#if _LIBCUDACXX_STD_VER < 20
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(
    default_sentinel_t, const counted_iterator& __lhs)
  {
    return __lhs.__count_ == 0;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(
    const counted_iterator& __lhs, default_sentinel_t)
  {
    return __lhs.__count_ != 0;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(
    default_sentinel_t, const counted_iterator& __lhs)
  {
    return __lhs.__count_ != 0;
  }
#endif

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr strong_ordering operator<=>(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ <=> __rhs.__count_;
  }
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator<(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ < __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator<(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ < __rhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator<=(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ <= __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator<=(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ <= __rhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator>(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ > __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator>(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ > __rhs.__count_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator>=(
    const counted_iterator& __lhs, const counted_iterator& __rhs)
  {
    return __lhs.__count_ >= __rhs.__count_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2)
    (requires common_with<_I2, _Iter>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator>=(
    const counted_iterator& __lhs, const counted_iterator<_I2>& __rhs)
  {
    return __lhs.__count_ >= __rhs.__count_;
  }

  template<class _I2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr auto iter_move(const counted_iterator<_I2>& __i)
    noexcept(noexcept(_CUDA_VRANGES::iter_move(__i.__current_)))
    _LIBCUDACXX_TRAILING_REQUIRES(iter_rvalue_reference_t<_I2>)(requires same_as<_I2, _Iter> && input_iterator<_I2>)
  {
    _LIBCUDACXX_ASSERT(__i.__count_ > 0, "Iterator must not be past end of range.");
    return _CUDA_VRANGES::iter_move(__i.__current_);
  }

  template<class _I2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr auto iter_swap(const counted_iterator& __x, const counted_iterator<_I2>& __y)
    noexcept(noexcept(_CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_)))
    _LIBCUDACXX_TRAILING_REQUIRES(void)(requires indirectly_swappable<_I2, _Iter>)
  {
    _LIBCUDACXX_ASSERT(__x.__count_ > 0 && __y.__count_ > 0,
                   "Iterators must not be past end of range.");
    return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(counted_iterator);

#if _LIBCUDACXX_STD_VER > 17
template<input_iterator _Iter>
  requires same_as<_ITER_TRAITS<_Iter>, iterator_traits<_Iter>>
struct iterator_traits<counted_iterator<_Iter>> : iterator_traits<_Iter> {
  using pointer = conditional_t<contiguous_iterator<_Iter>,
                                add_pointer_t<iter_reference_t<_Iter>>, void>;
};
#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv
template<class _Iter>
struct iterator_traits<counted_iterator<_Iter>, enable_if_t<input_iterator<_Iter> &&
                                                            same_as<_ITER_TRAITS<_Iter>, iterator_traits<_Iter>>>>
  : iterator_traits<_Iter> {
  using pointer = conditional_t<contiguous_iterator<_Iter>,
                                add_pointer_t<iter_reference_t<_Iter>>, void>;
};

// In C++17 we end up in an infinite recursion trying to determine the return type of `to_address`
template <class _Iter>
struct pointer_traits<counted_iterator<_Iter>, enable_if_t<contiguous_iterator<_Iter>>> {
    using pointer         = counted_iterator<_Iter>;
    using element_type    = typename pointer_traits<_Iter>::element_type;
    using difference_type = typename pointer_traits<_Iter>::difference_type;

    _LIBCUDACXX_HIDE_FROM_ABI inline _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr auto to_address(const pointer __iter) noexcept {
      return _CUDA_VSTD::to_address(__iter.base());
    }
};

#endif // _LIBCUDACXX_STD_VER < 20

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(pop)
#endif // _LIBCUDACXX_COMPILER_MSVC

#endif // _LIBCUDACXX___ITERATOR_COUNTED_ITERATOR_H
