// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_IOTA_ITERATOR_H
#define _LIBCUDACXX___RANGES_IOTA_ITERATOR_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Int>
struct __get_wider_signed
{
  _LIBCUDACXX_HIDE_FROM_ABI static auto __call() noexcept
  {
    if constexpr (sizeof(_Int) < sizeof(short))
    {
      return type_identity<short>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(int))
    {
      return type_identity<int>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(long))
    {
      return type_identity<long>{};
    }
    else
    {
      return type_identity<long long>{};
    }

    static_assert(sizeof(_Int) <= sizeof(long long),
                  "Found integer-like type that is bigger than largest integer like type.");
    _CCCL_UNREACHABLE();
  }

  using type = typename decltype(__call())::type;
};

template <class _Start>
using _IotaDiffT =
  typename conditional_t<(!integral<_Start> || sizeof(iter_difference_t<_Start>) > sizeof(_Start)),
                         type_identity<iter_difference_t<_Start>>,
                         __get_wider_signed<_Start>>::type;

template <class _Iter>
_CCCL_CONCEPT __decrementable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(incrementable<_Iter>), _Same_as(_Iter&)(--__i), _Same_as(_Iter)(__i--));

template <class _Iter>
_CCCL_CONCEPT __advanceable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i, const _Iter __j, const _IotaDiffT<_Iter> __n)(
  requires(__decrementable<_Iter>),
  requires(totally_ordered<_Iter>),
  _Same_as(_Iter&) __i += __n,
  _Same_as(_Iter&) __i -= __n,
  requires(is_constructible_v<_Iter, decltype(__j + __n)>),
  requires(is_constructible_v<_Iter, decltype(__n + __j)>),
  requires(is_constructible_v<_Iter, decltype(__j - __n)>),
  requires(convertible_to<decltype(__j - __j), _IotaDiffT<_Iter>>));

template <class, class = void>
struct __iota_iterator_category
{};

template <class _Tp>
struct __iota_iterator_category<_Tp, enable_if_t<incrementable<_Tp>>>
{
  using iterator_category = input_iterator_tag;
};

#if !defined(_CCCL_NO_CONCEPTS)
template <weakly_incrementable _Start>
  requires copyable<_Start>
#else // ^^^ !_CCCL_NO_CONCEPTS ^^^ / vvv _CCCL_NO_CONCEPTS vvv
template <class _Start, enable_if_t<weakly_incrementable<_Start>, int> = 0, enable_if_t<copyable<_Start>, int> = 0>
#endif // _CCCL_NO_CONCEPTS
struct __iota_iterator : public __iota_iterator_category<_Start>
{
  using iterator_concept =
    conditional_t<__advanceable<_Start>,
                  random_access_iterator_tag,
                  conditional_t<__decrementable<_Start>,
                                bidirectional_iterator_tag,
                                conditional_t<incrementable<_Start>,
                                              forward_iterator_tag,
                                              /*Else*/ input_iterator_tag>>>;

  using value_type      = _Start;
  using difference_type = _IotaDiffT<_Start>;

  _Start __value_ = _Start();

#if !defined(_CCCL_NO_CONCEPTS)
  _CCCL_HIDE_FROM_ABI __iota_iterator()
    requires default_initializable<_Start>
  = default;
#else // ^^^ !_CCCL_NO_CONCEPTS ^^^ / vvv _CCCL_NO_CONCEPTS vvv
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(default_initializable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator() noexcept(is_nothrow_default_constructible_v<_Start2>) {}
#endif // _CCCL_NO_CONCEPTS

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __iota_iterator(_Start __value) noexcept(
    is_nothrow_move_constructible_v<_Start>)
      : __value_(_CUDA_VSTD::move(__value))
  {}

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Start operator*() const
    noexcept(is_nothrow_copy_constructible_v<_Start>)
  {
    return __value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Start2 operator[](difference_type __n) const
    noexcept(is_nothrow_copy_constructible_v<_Start2> && noexcept(_CUDA_VSTD::declval<const _Start2&>() + __n))
  {
    if constexpr (__integer_like<_Start>)
    {
      return _Start(__value_ + static_cast<_Start>(__n));
    }
    else
    {
      return _Start(__value_ + __n);
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator& operator++() noexcept(noexcept(++_CUDA_VSTD::declval<_Start&>()))
  {
    ++__value_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator++(int) noexcept(noexcept(++_CUDA_VSTD::declval<_Start&>()) && is_nothrow_copy_constructible_v<_Start>)
  {
    if constexpr (incrementable<_Start>)
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
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator& operator--() noexcept(noexcept(--_CUDA_VSTD::declval<_Start2&>()))
  {
    --__value_;
    return *this;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator
  operator--(int) noexcept(noexcept(--_CUDA_VSTD::declval<_Start2&>()) && is_nothrow_copy_constructible_v<_Start>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator& operator+=(difference_type __n) noexcept(__integer_like<_Start2>)
  {
    if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>)
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
    else if constexpr (__signed_integer_like<_Start>)
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
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iota_iterator& operator-=(difference_type __n) noexcept(__integer_like<_Start2>)
  {
    if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>)
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
    else if constexpr (__signed_integer_like<_Start>)
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
  _CCCL_REQUIRES(equality_comparable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() == _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ == __y.__value_;
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(equality_comparable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() != _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ != __y.__value_;
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __x.__value_ < __y.__value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return __y < __x;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<=(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__y < __x);
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(totally_ordered<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>=(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() < _CUDA_VSTD::declval<const _Start2&>()))
  {
    return !(__x < __y);
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Start2&>() <=> _CUDA_VSTD::declval<const _Start2&>()))
    requires totally_ordered<_Start> && three_way_comparable<_Start>
  {
    return __x.__value_ <=> __y.__value_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iota_iterator
  operator+(__iota_iterator __i, difference_type __n) noexcept(__integer_like<_Start2>)
  {
    __i += __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iota_iterator
  operator+(difference_type __n, __iota_iterator __i) noexcept(__integer_like<_Start2>)
  {
    return __i + __n;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iota_iterator
  operator-(__iota_iterator __i, difference_type __n) noexcept(__integer_like<_Start2>)
  {
    __i -= __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iota_iterator& __x, const __iota_iterator& __y) noexcept(__integer_like<_Start2>)
  {
    if constexpr (__integer_like<_Start> && !__signed_integer_like<_Start>)
    {
      if (__y.__value_ > __x.__value_)
      {
        return static_cast<difference_type>(-static_cast<difference_type>(__y.__value_ - __x.__value_));
      }
      return static_cast<difference_type>(__x.__value_ - __y.__value_);
    }
    else if constexpr (__signed_integer_like<_Start>)
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

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_IOTA_ITERATOR_H
