//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Int>
struct __get_wider_signed
{
  _CCCL_API inline static auto __call() noexcept
  {
    if constexpr (sizeof(_Int) < sizeof(short))
    {
      return ::cuda::std::type_identity<short>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(int))
    {
      return ::cuda::std::type_identity<int>{};
    }
    else if constexpr (sizeof(_Int) < sizeof(long))
    {
      return ::cuda::std::type_identity<long>{};
    }
    else
    {
      return ::cuda::std::type_identity<long long>{};
    }

    static_assert(sizeof(_Int) <= sizeof(long long),
                  "Found integer-like type that is bigger than largest integer like type.");
    _CCCL_UNREACHABLE();
  }

  using type = typename decltype(__call())::type;
};

template <class _Start>
using _IotaDiffT = typename ::cuda::std::conditional_t<
  (!::cuda::std::integral<_Start> || sizeof(::cuda::std::iter_difference_t<_Start>) > sizeof(_Start)),
  ::cuda::std::type_identity<::cuda::std::iter_difference_t<_Start>>,
  __get_wider_signed<_Start>>::type;

template <class _Iter>
_CCCL_CONCEPT __decrementable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(::cuda::std::incrementable<_Iter>), _Same_as(_Iter&)(--__i), _Same_as(_Iter)(__i--));

template <class _Iter>
_CCCL_CONCEPT __advanceable = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i, const _Iter __j, const _IotaDiffT<_Iter> __n)(
  requires(__decrementable<_Iter>),
  requires(::cuda::std::totally_ordered<_Iter>),
  _Same_as(_Iter&) __i += __n,
  _Same_as(_Iter&) __i -= __n,
  requires(::cuda::std::is_constructible_v<_Iter, decltype(__j + __n)>),
  requires(::cuda::std::is_constructible_v<_Iter, decltype(__n + __j)>),
  requires(::cuda::std::is_constructible_v<_Iter, decltype(__j - __n)>),
  requires(::cuda::std::convertible_to<decltype(__j - __j), _IotaDiffT<_Iter>>));

template <class, class = void>
struct __counting_iterator_category
{};

template <class _Tp>
struct __counting_iterator_category<_Tp, ::cuda::std::enable_if_t<::cuda::std::incrementable<_Tp>>>
{
  using iterator_category = ::cuda::std::input_iterator_tag;
};

//! @brief \p counting_iterator is an iterator which represents a pointer into a range of sequentially changing values.
//! This iterator is useful for creating a range filled with a sequence without explicitly storing it in memory. Using
//! \p counting_iterator saves memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a \p counting_iterator whose \c value_type is \c int and which
//! sequentially increments by \c 1.
//!
//! @code
//! #include <cuda/iterator>
//! ...
//! // create iterators
//! cuda::counting_iterator first(10);
//! cuda::counting_iterator last = first + 3;
//!
//! first[0]   // returns 10
//! first[1]   // returns 11
//! first[100] // returns 110
//!
//! // sum of [first, last)
//! thrust::reduce(first, last);   // returns 33 (i.e. 10 + 11 + 12)
//!
//! // initialize vector to [0,1,2,..]
//! cuda::counting_iterator iter(0);
//! thrust::device_vector<int> vec(500);
//! thrust::copy(iter, iter + vec.size(), vec.begin());
//! @endcode
//!
//! This next example demonstrates how to use a \p counting_iterator with the \p thrust::copy_if function to compute the
//! indices of the non-zero elements of a \p device_vector. In this example, we use the \p make_counting_iterator
//! function to avoid specifying the type of the \p counting_iterator.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/copy.h>
//! #include <thrust/functional.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!  // this example computes indices for all the nonzero values in a sequence
//!
//!  // sequence of zero and nonzero values
//!  thrust::device_vector<int> stencil{0, 1, 1, 0, 0, 1, 0, 1};
//!
//!  // storage for the nonzero indices
//!  thrust::device_vector<int> indices(8);
//!
//!  // use make_counting_iterator to define the sequence [0, 8)
//!  auto indices_end = thrust::copy_if(cuda::counting_iterator{0},
//!                                     cuda::counting_iterator{8},
//!                                     stencil.begin(),
//!                                     indices.begin(),
//!                                     ::cuda::std::identity{});
//!  // indices now contains [1,2,5,7]
//!
//!  return 0;
//! }
//! @endcode
#if _CCCL_HAS_CONCEPTS()
template <::cuda::std::weakly_incrementable _Start>
  requires ::cuda::std::copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          ::cuda::std::enable_if_t<::cuda::std::weakly_incrementable<_Start>, int> = 0,
          ::cuda::std::enable_if_t<::cuda::std::copyable<_Start>, int>             = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
struct counting_iterator : public __counting_iterator_category<_Start>
{
  using iterator_concept = ::cuda::std::conditional_t<
    __advanceable<_Start>,
    ::cuda::std::random_access_iterator_tag,
    ::cuda::std::conditional_t<__decrementable<_Start>,
                               ::cuda::std::bidirectional_iterator_tag,
                               ::cuda::std::conditional_t<::cuda::std::incrementable<_Start>,
                                                          ::cuda::std::forward_iterator_tag,
                                                          /*Else*/ ::cuda::std::input_iterator_tag>>>;

  using value_type      = _Start;
  using difference_type = _IotaDiffT<_Start>;

  _Start __value_ = _Start();

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI counting_iterator()
    requires ::cuda::std::default_initializable<_Start>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Start2>)
  _CCCL_API constexpr counting_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Start2>) {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_API constexpr explicit counting_iterator(_Start __value) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Start>)
      : __value_(::cuda::std::move(__value))
  {}

  [[nodiscard]] _CCCL_API constexpr _Start operator*() const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Start>)
  {
    return __value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API constexpr _Start2 operator[](difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Start2>
             && noexcept(::cuda::std::declval<const _Start2&>() + __n))
  {
    if constexpr (::cuda::std::__integer_like<_Start>)
    {
      return _Start(__value_ + static_cast<_Start>(__n));
    }
    else
    {
      return _Start(__value_ + __n);
    }
  }

  _CCCL_API constexpr counting_iterator& operator++() noexcept(noexcept(++::cuda::std::declval<_Start&>()))
  {
    ++__value_;
    return *this;
  }

  _CCCL_API constexpr auto operator++(int) noexcept(
    noexcept(++::cuda::std::declval<_Start&>()) && ::cuda::std::is_nothrow_copy_constructible_v<_Start>)
  {
    if constexpr (::cuda::std::incrementable<_Start>)
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
  _CCCL_API constexpr counting_iterator& operator--() noexcept(noexcept(--::cuda::std::declval<_Start2&>()))
  {
    --__value_;
    return *this;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__decrementable<_Start2>)
  _CCCL_API constexpr counting_iterator operator--(int) noexcept(
    noexcept(--::cuda::std::declval<_Start2&>()) && ::cuda::std::is_nothrow_copy_constructible_v<_Start>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  _CCCL_API constexpr counting_iterator& operator+=(difference_type __n) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    if constexpr (::cuda::std::__integer_like<_Start> && !::cuda::std::__signed_integer_like<_Start>)
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
    else if constexpr (::cuda::std::__signed_integer_like<_Start>)
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
  _CCCL_API constexpr counting_iterator& operator-=(difference_type __n) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    if constexpr (::cuda::std::__integer_like<_Start> && !::cuda::std::__signed_integer_like<_Start>)
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
    else if constexpr (::cuda::std::__signed_integer_like<_Start>)
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
  _CCCL_REQUIRES(::cuda::std::equality_comparable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() == ::cuda::std::declval<const _Start2&>()))
  {
    return __x.__value_ == __y.__value_;
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::equality_comparable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() != ::cuda::std::declval<const _Start2&>()))
  {
    return __x.__value_ != __y.__value_;
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() < ::cuda::std::declval<const _Start2&>()))
  {
    return __x.__value_ < __y.__value_;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() < ::cuda::std::declval<const _Start2&>()))
  {
    return __y < __x;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() < ::cuda::std::declval<const _Start2&>()))
  {
    return !(__y < __x);
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(::cuda::std::totally_ordered<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() < ::cuda::std::declval<const _Start2&>()))
  {
    return !(__x < __y);
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const counting_iterator& __x, const counting_iterator& __y) noexcept(
    noexcept(::cuda::std::declval<const _Start2&>() <=> ::cuda::std::declval<const _Start2&>()))
    requires ::cuda::std::totally_ordered<_Start> && ::cuda::std::three_way_comparable<_Start>
  {
    return __x.__value_ <=> __y.__value_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator+(counting_iterator __i, difference_type __n) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    __i += __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator+(difference_type __n, counting_iterator __i) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    return __i + __n;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr counting_iterator
  operator-(counting_iterator __i, difference_type __n) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    __i -= __n;
    return __i;
  }

  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(__advanceable<_Start2>)
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const counting_iterator& __x, const counting_iterator& __y) noexcept(::cuda::std::__integer_like<_Start2>)
  {
    if constexpr (::cuda::std::__integer_like<_Start> && !::cuda::std::__signed_integer_like<_Start>)
    {
      if (__y.__value_ > __x.__value_)
      {
        return static_cast<difference_type>(-static_cast<difference_type>(__y.__value_ - __x.__value_));
      }
      return static_cast<difference_type>(__x.__value_ - __y.__value_);
    }
    else if constexpr (::cuda::std::__signed_integer_like<_Start>)
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
[[nodiscard]] _CCCL_API constexpr auto make_counting_iterator(_Start __start)
{
  return counting_iterator<_Start>{__start};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_COUNTING_ITERATOR_H
