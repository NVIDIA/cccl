//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_DISCARD_ITERATOR_H
#define _CUDA___ITERATOR_DISCARD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief \p discard_iterator is an iterator which represents a special kind of pointer that ignores values written to
//! it upon dereference. This iterator is useful for ignoring the output of certain algorithms without wasting memory
//! capacity or bandwidth. \p discard_iterator may also be used to count the size of an algorithm's output which may not
//! be known a priori.
//!
//! The following code snippet demonstrates how to use \p discard_iterator to ignore one of the output ranges of
//! reduce_by_key
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/reduce.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> keys(7), values(7);
//!
//!   keys[0] = 1;
//!   keys[1] = 3;
//!   keys[2] = 3;
//!   keys[3] = 3;
//!   keys[4] = 2;
//!   keys[5] = 2;
//!   keys[6] = 1;
//!
//!   values[0] = 9;
//!   values[1] = 8;
//!   values[2] = 7;
//!   values[3] = 6;
//!   values[4] = 5;
//!   values[5] = 4;
//!   values[6] = 3;
//!
//!   thrust::device_vector<int> result(4);
//!
//!   // we are only interested in the reduced values
//!   // use discard_iterator to ignore the output keys
//!   thrust::reduce_by_key(keys.begin(), keys.end(),
//!                         values.begin(),
//!                         cuda::make_discard_iterator(),
//!                         result.begin());
//!
//!   // result is now [9, 21, 9, 3]
//!
//!   return 0;
//! }
//! @endcode
class discard_iterator
{
private:
  _CUDA_VSTD::ptrdiff_t __counter_ = 0;

  struct __discard_proxy
  {
    _CCCL_TEMPLATE(class _Tp)
    _CCCL_REQUIRES((!_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Tp>, __discard_proxy>) )
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __discard_proxy& operator=(_Tp&&) noexcept
    {
      return *this;
    }
  };

public:
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using value_type        = void;
  using pointer           = void*;
  using reference         = void;

  //! @brief Default constructs a \p discard_iterator with a value initialized counter
  _CCCL_HIDE_FROM_ABI constexpr discard_iterator() = default;

  //! @brief Constructs a \p discard_iterator with a given \p __counter
  //! @param __counter The counter used for the discard iterator
  _CCCL_TEMPLATE(class _Integer)
  _CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Integer>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator(_Integer __counter) noexcept
      : __counter_(static_cast<_CUDA_VSTD::ptrdiff_t>(__counter))
  {}

  //! @brief Returns the stored counter
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr difference_type count() const noexcept
  {
    return __counter_;
  }

  //! @brief Dereferences the \c discard_iterator returning a proxy that discards all values that are assigned to it
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr __discard_proxy operator*() const noexcept
  {
    return {};
  }

  //! @brief Subscipts the \c discard_iterator returning a proxy that discards all values that are assigned to it
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr __discard_proxy operator[](difference_type) const noexcept
  {
    return {};
  }

  //! @brief Increments the stored counter
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator& operator++() noexcept
  {
    ++__counter_;
    return *this;
  }

  //! @brief Increments the stored counter
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator operator++(int) noexcept
  {
    discard_iterator __tmp = *this;
    ++__counter_;
    return __tmp;
  }

  //! @brief Decrements the stored counter
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator& operator--() noexcept
  {
    --__counter_;
    return *this;
  }

  //! @brief Decrements the stored counter
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator operator--(int) noexcept
  {
    discard_iterator __tmp = *this;
    --__counter_;
    return __tmp;
  }

  //! @brief Returns a copy of this \c discard_iterator advanced by \p __n
  //! @param __n The number of elements to advance
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator operator+(difference_type __n) const noexcept
  {
    return discard_iterator{__counter_ + __n};
  }

  //! @brief Returns a copy of \p __x advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __x The original \c discard_iterator
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr discard_iterator
  operator+(difference_type __n, const discard_iterator& __x) noexcept
  {
    return __x + __n;
  }

  //! @brief Advances this \c discard_iterator by \p __n
  //! @param __n The number of elements to advance
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator& operator+=(difference_type __n) noexcept
  {
    __counter_ += __n;
    return *this;
  }

  //! @brief Returns a copy of this \c discard_iterator decremented by \p __n
  //! @param __n The number of elements to decrement
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator operator-(difference_type __n) const noexcept
  {
    return discard_iterator{__counter_ - __n};
  }

  //! @brief Returns the distance between \p __lhs and \p __rhs
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return __rhs.__counter_ - __lhs.__counter_
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__counter_ - __lhs.__counter_;
  }

  //! @brief Returns the distance between \p __lhs a \p default_sentinel
  //! @param __lhs The left \c discard_iterator
  //! @return -__lhs.__counter_
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const discard_iterator& __lhs, _CUDA_VSTD::default_sentinel_t) noexcept
  {
    return static_cast<difference_type>(-__lhs.__counter_);
  }

  //! @brief Returns the distance between a \p default_sentinel and \p __rhs
  //! @param __rhs The right \c discard_iterator
  //! @return __rhs.__coutner_
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(_CUDA_VSTD::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return static_cast<difference_type>(__rhs.__counter_);
  }

  //! @brief Decrements the \c discard_iterator by \p __n
  //! @param __n The number of elements to decrement
  _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator& operator-=(difference_type __n) noexcept
  {
    __counter_ -= __n;
    return *this;
  }

  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for equality
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if both iterators store the same counter
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ == __rhs.__counter_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for inequality
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if both iterators store different counters
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ != __rhs.__counter_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares a \c discard_iterator \p __lhs with \p default_sentinel for equality
  //! @param __lhs The left \c discard_iterator
  //! @return true if the counter of \p __lhs is zero
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const discard_iterator& __lhs, _CUDA_VSTD::default_sentinel_t) noexcept
  {
    return __lhs.__counter_ == 0;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares a \c discard_iterator \p __rhs with \p default_sentinel for equality
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter of \p __rhs is zero
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(_CUDA_VSTD::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__counter_ == 0;
  }

  //! @brief Compares a \c discard_iterator \p __rhs with \p default_sentinel for inequality
  //! @param __lhs The right \c discard_iterator
  //! @return true if the counter of \p __lhs is not zero
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const discard_iterator& __lhs, _CUDA_VSTD::default_sentinel_t) noexcept
  {
    return __lhs.__counter_ != 0;
  }

  //! @brief Compares a \c discard_iterator \p __rhs with \p default_sentinel for inequality
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter of \p __rhs is not zero
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(_CUDA_VSTD::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__counter_ != 0;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c discard_iterator \p __lhs and \p __rhs
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return the three-way ordering of the counters stored by \p __lhs and \p __rhs
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr strong_ordering
  operator<=>(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ <=> __rhs.__counter_;
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for less than
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter stored by \p __lhs compares less than the one stored by \p __rhs
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ < __rhs.__counter_;
  }

  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for less equal
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter stored by \p __lhs compares less equal the one stored by \p __rhs
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ <= __rhs.__counter_;
  }

  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for greater than
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter stored by \p __lhs compares less greater the one stored by \p __rhs
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ > __rhs.__counter_;
  }

  //! @brief Compares two \c discard_iterator \p __lhs and \p __rhs for greater equal
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return true if the counter stored by \p __lhs compares greater equal the one stored by \p __rhs
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__counter_ >= __rhs.__counter_;
  }
};

//! @brief Creates a \p discard_iterator from an optional counter.
//! @param __counter The index of the returned \p discard_iterator within a range. In the default case, the value of
//! this parameter is \c 0.
//! @return A new \p discard_iterator with \p __counter as the couner.
_CCCL_TEMPLATE(class _Integer = _CUDA_VSTD::ptrdiff_t)
_CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_Integer>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr discard_iterator make_discard_iterator(_Integer __counter = 0)
{
  return discard_iterator{__counter};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_DISCARD_ITERATOR_H
