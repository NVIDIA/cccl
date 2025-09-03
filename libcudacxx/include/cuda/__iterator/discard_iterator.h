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

_CCCL_BEGIN_NAMESPACE_CUDA

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
//!   thrust::device_vector<int> keys{1, 3, 3, 3, 2, 2, 1};
//!   thrust::device_vector<int> values{9, 8, 7, 6, 5, 4, 3};
//!
//!   thrust::device_vector<int> result(4);
//!
//!   // we are only interested in the reduced values
//!   // use discard_iterator to ignore the output keys
//!   thrust::reduce_by_key(keys.begin(), keys.end(),
//!                         values.begin(),
//!                         cuda::discard_iterator{},
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
  ::cuda::std::ptrdiff_t __index_ = 0;

public:
  struct __discard_proxy
  {
    _CCCL_TEMPLATE(class _Tp)
    _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Tp>, __discard_proxy>) )
    _CCCL_API constexpr const __discard_proxy& operator=(_Tp&&) const noexcept
    {
      return *this;
    }
  };

  using iterator_concept  = ::cuda::std::random_access_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using difference_type   = ::cuda::std::ptrdiff_t;
  using value_type        = void;
  using pointer           = void;
  using reference         = void;

  //! @brief Default constructs a \p discard_iterator at index zero
  _CCCL_HIDE_FROM_ABI constexpr discard_iterator() = default;

  //! @brief Constructs a \p discard_iterator with a given \p __index
  //! @param __index The index used for the discard iterator
  _CCCL_TEMPLATE(class _Integer)
  _CCCL_REQUIRES(::cuda::std::__integer_like<_Integer>)
  _CCCL_API constexpr discard_iterator(_Integer __index) noexcept
      : __index_(static_cast<::cuda::std::ptrdiff_t>(__index))
  {}

  //! @brief Returns the stored index
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return __index_;
  }

  //! @brief Dereferences the \c discard_iterator returning a proxy that discards all values that are assigned to it
  [[nodiscard]] _CCCL_API constexpr __discard_proxy operator*() const noexcept
  {
    return {};
  }

  //! @brief Subscipts the \c discard_iterator returning a proxy that discards all values that are assigned to it
  [[nodiscard]] _CCCL_API constexpr __discard_proxy operator[](difference_type) const noexcept
  {
    return {};
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr discard_iterator& operator++() noexcept
  {
    ++__index_;
    return *this;
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr discard_iterator operator++(int) noexcept
  {
    discard_iterator __tmp = *this;
    ++__index_;
    return __tmp;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr discard_iterator& operator--() noexcept
  {
    --__index_;
    return *this;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr discard_iterator operator--(int) noexcept
  {
    discard_iterator __tmp = *this;
    --__index_;
    return __tmp;
  }

  //! @brief Returns a copy of this \c discard_iterator advanced by \p __n
  //! @param __n The number of elements to advance
  [[nodiscard]] _CCCL_API constexpr discard_iterator operator+(difference_type __n) const noexcept
  {
    return discard_iterator{__index_ + __n};
  }

  //! @brief Returns a copy of \p __x advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __x The original \c discard_iterator
  [[nodiscard]] _CCCL_API friend constexpr discard_iterator
  operator+(difference_type __n, const discard_iterator& __x) noexcept
  {
    return __x + __n;
  }

  //! @brief Advances the index of this \c discard_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_API constexpr discard_iterator& operator+=(difference_type __n) noexcept
  {
    __index_ += __n;
    return *this;
  }

  //! @brief Returns a copy of this \c discard_iterator decremented by \p __n
  //! @param __n The number of elements to decrement
  [[nodiscard]] _CCCL_API constexpr discard_iterator operator-(difference_type __n) const noexcept
  {
    return discard_iterator{__index_ - __n};
  }

  //! @brief Returns the distance between \p __lhs and \p __rhs
  //! @param __lhs The left \c discard_iterator
  //! @param __rhs The right \c discard_iterator
  //! @return __rhs.__index_ - __lhs.__index_
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__index_ - __lhs.__index_;
  }

  //! @brief Returns the distance between \p __lhs a \p default_sentinel
  //! @param __lhs The left \c discard_iterator
  //! @return -__lhs.__index_
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const discard_iterator& __lhs, ::cuda::std::default_sentinel_t) noexcept
  {
    return static_cast<difference_type>(-__lhs.__index_);
  }

  //! @brief Returns the distance between a \p default_sentinel and \p __rhs
  //! @param __rhs The right \c discard_iterator
  //! @return __rhs.__index_
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(::cuda::std::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return static_cast<difference_type>(__rhs.__index_);
  }

  //! @brief Decrements the index of the \c discard_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_API constexpr discard_iterator& operator-=(difference_type __n) noexcept
  {
    __index_ -= __n;
    return *this;
  }

  //! @brief Compares two \c discard_iterator for equality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ == __rhs.__index_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c discard_iterator for inequality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ != __rhs.__index_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares a \c discard_iterator with \p default_sentinel , true if the index of \p __lhs is zero
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const discard_iterator& __lhs, ::cuda::std::default_sentinel_t) noexcept
  {
    return __lhs.__index_ == 0;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares a \c discard_iterator with \p default_sentinel , true if the index of \p __lhs is zero
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(::cuda::std::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__index_ == 0;
  }

  //! @brief Compares a \c discard_iterator with \p default_sentinel , true if the index of \p __lhs is not zero
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const discard_iterator& __lhs, ::cuda::std::default_sentinel_t) noexcept
  {
    return __lhs.__index_ != 0;
  }

  //! @brief Compares a \c discard_iterator with \p default_sentinel , true if the index of \p __lhs is not zero
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(::cuda::std::default_sentinel_t, const discard_iterator& __rhs) noexcept
  {
    return __rhs.__index_ != 0;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c discard_iterator by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering
  operator<=>(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <=> __rhs.__index_;
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Compares two \c discard_iterator for less than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ < __rhs.__index_;
  }

  //! @brief Compares two \c discard_iterator for less equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <= __rhs.__index_;
  }

  //! @brief Compares two \c discard_iterator for greater than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ > __rhs.__index_;
  }

  //! @brief Compares two \c discard_iterator for greater equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const discard_iterator& __lhs, const discard_iterator& __rhs) noexcept
  {
    return __lhs.__index_ >= __rhs.__index_;
  }
};

//! @brief Creates a \p discard_iterator from an optional index.
//! @param __index The index of the \p discard_iterator within a range. The default index is \c 0.
//! @return A new \p discard_iterator with \p __index as the couner.
_CCCL_TEMPLATE(class _Integer = ::cuda::std::ptrdiff_t)
_CCCL_REQUIRES(::cuda::std::__integer_like<_Integer>)
[[nodiscard]] _CCCL_API constexpr discard_iterator make_discard_iterator(_Integer __index = 0)
{
  return discard_iterator{__index};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_DISCARD_ITERATOR_H
