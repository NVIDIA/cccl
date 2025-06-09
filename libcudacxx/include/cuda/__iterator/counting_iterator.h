// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_COUNTING_ITERATOR_H
#define _CUDA___ITERATOR_COUNTING_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/unreachable_sentinel.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Int>
struct __get_wider_signed
{
  _LIBCUDACXX_HIDE_FROM_ABI static auto __call() noexcept
  {
    if constexpr (sizeof(_Int) < sizeof(short))
    {
      return _CUDA_VSTD::type_identity<short>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(int))
    {
      return _CUDA_VSTD::type_identity<int>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(long))
    {
      return _CUDA_VSTD::type_identity<long>{};
    }
    else
    {
      return _CUDA_VSTD::type_identity<long long>{};
    }

    static_assert(sizeof(_Int) <= sizeof(long long),
                  "Found integer-like type that is bigger than largest integer like type.");
    _CCCL_UNREACHABLE();
  }

  using type = typename decltype(__call())::type;
};

template <class _Start>
using _IotaDiffT = typename _CUDA_VSTD::conditional_t<
  (!_CUDA_VSTD::integral<_Start> || sizeof(_CUDA_VSTD::iter_difference_t<_Start>) > sizeof(_Start)),
  _CUDA_VSTD::type_identity<_CUDA_VSTD::iter_difference_t<_Start>>,
  __get_wider_signed<_Start>>::type;

template <class _Iter>
_CCCL_CONCEPT __decrementable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(_CUDA_VSTD::incrementable<_Iter>), _Same_as(_Iter&)(--__i), _Same_as(_Iter)(__i--));

template <class _Iter>
_CCCL_CONCEPT __advanceable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i, const _Iter __j, const _IotaDiffT<_Iter> __n)(
  requires(__decrementable<_Iter>),
  requires(_CUDA_VSTD::totally_ordered<_Iter>),
  _Same_as(_Iter&) __i += __n,
  _Same_as(_Iter&) __i -= __n,
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__j + __n)>),
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__n + __j)>),
  requires(_CUDA_VSTD::is_constructible_v<_Iter, decltype(__j - __n)>),
  requires(_CUDA_VSTD::convertible_to<decltype(__j - __j), _IotaDiffT<_Iter>>));

template <class, class = void>
struct __counting_iterator_category
{};

template <class _Tp>
struct __counting_iterator_category<_Tp, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::incrementable<_Tp>>>
{
  using iterator_category = _CUDA_VSTD::input_iterator_tag;
};

#if _CCCL_HAS_CONCEPTS()
template <_CUDA_VSTD::weakly_incrementable _Start>
  requires _CUDA_VSTD::copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          _CUDA_VSTD::enable_if_t<_CUDA_VSTD::weakly_incrementable<_Start>, int> = 0,
          _CUDA_VSTD::enable_if_t<_CUDA_VSTD::copyable<_Start>, int>             = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
struct counting_iterator : public __counting_iterator_category<_Start>
{
  using iterator_concept = _CUDA_VSTD::conditional_t<
    __advanceable<_Start>,
    _CUDA_VSTD::random_access_iterator_tag,
    _CUDA_VSTD::conditional_t<__decrementable<_Start>,
                              _CUDA_VSTD::bidirectional_iterator_tag,
                              _CUDA_VSTD::conditional_t<_CUDA_VSTD::incrementable<_Start>,
                                                        _CUDA_VSTD::forward_iterator_tag,
                                                        /*Else*/ _CUDA_VSTD::input_iterator_tag>>>;

  using value_type      = _Start;
  using difference_type = _IotaDiffT<_Start>;

  _Start __value_ = _Start();

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI counting_iterator()
    requires _CUDA_VSTD::default_initializable<_Start>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::default_initializable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator() noexcept(
    _CUDA_VSTD::is_nothrow_default_constructible_v<_Start2>)
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit counting_iterator(_Start __value) noexcept(
    _CUDA_VSTD::is_nothrow_move_constructible_v<_Start>)
      : __value_(_CUDA_VSTD::move(__value))
  {}

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Start operator*() const
    noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    return __value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Start2 operator[](difference_type __n) const
    noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Start2>
             && noexcept(_CUDA_VSTD::declval<const _Start2&>() + __n))
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start>)
    {
      return _Start(__value_ + static_cast<_Start>(__n));
    }
    else
    {
      return _Start(__value_ + __n);
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator&
  operator++() noexcept(noexcept(++_CUDA_VSTD::declval<_Start&>()))
  {
    ++__value_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator++(int) noexcept(
    noexcept(++_CUDA_VSTD::declval<_Start&>()) && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    if constexpr (_CUDA_VSTD::incrementable<_Start>)
    {
      auto __tmp = *this;
      ++__value_;
      return __tmp;
    }
    else
    {
      ++__value_;
    }
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator&
  operator--() noexcept(noexcept(--_CUDA_VSTD::declval<_Start2&>()))
  {
    --__value_;
    return *this;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator operator--(int) noexcept(
    noexcept(--_CUDA_VSTD::declval<_Start2&>()) && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Start>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator&
  operator+=(difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__n >= difference_type(0))
      {
        __value_ += static_cast<_Start>(__n);
      }
      else
      {
        __value_ -= static_cast<_Start>(-__n);
      }
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      __value_ += static_cast<_Start>(__n);
    }
    else
    {
      __value_ += __n;
    }
    return *this;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_iterator&
  operator-=(difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__n >= difference_type(0))
      {
        __value_ -= static_cast<_Start>(__n);
      }
      else
      {
        __value_ += static_cast<_Start>(-__n);
      }
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      __value_ -= static_cast<_Start>(__n);
    }
    else
    {
      __value_ -= __n;
    }
    return *this;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() == _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ == __y.__value_;
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() != _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ != __y.__value_;
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ < __y.__value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __y < __x;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__y < __x);
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__x < __y);
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() <=> _CUDA_VSTD::declval<const _Start2&>()))
    requires _CUDA_VSTD::totally_ordered<_Start> && _CUDA_VSTD::three_way_comparable<_Start>
  {
    return __x.__value_ <=> __y.__value_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr counting_iterator
  operator+(counting_iterator __i, difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    __i += __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr counting_iterator
  operator+(difference_type __n, counting_iterator __i) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    return __i + __n;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr counting_iterator
  operator-(counting_iterator __i, difference_type __n) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    __i -= __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const counting_iterator& __x, const counting_iterator& __y) noexcept(_CUDA_VSTD::__integer_like<_Start2>)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_Start> && !_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      if (__y.__value_ > __x.__value_)
      {
        return static_cast<difference_type>(-static_cast<difference_type>(__y.__value_ - __x.__value_));
      }
      return static_cast<difference_type>(__x.__value_ - __y.__value_);
    }
    else if constexpr (_CUDA_VSTD::__signed_integer_like<_Start>)
    {
      return static_cast<difference_type>(
        static_cast<difference_type>(__x.__value_) - static_cast<difference_type>(__y.__value_));
    }
    else
    {
      return __x.__value_ - __y.__value_;
    }
    _CCCL_UNREACHABLE();
  }
};

//! @brief make_counting_iterator creates a \p counting_iterator from an __integer-like__ \c _Start
//! @param __start The __integer-like__ \c _Start representing the initial count
template <class _Start>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto make_counting_iterator(_Start __start)
{
  return counting_iterator<_Start>{__start};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_COUNTING_ITERATOR_H
