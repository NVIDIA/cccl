//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_PERMUTATION_ITERATOR_H
#define _CUDA___ITERATOR_PERMUTATION_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/detail/libcxx/include/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @p permutation_iterator is an iterator which represents a pointer into a reordered view of a given range. \p
//! permutation_iterator is an imprecise name; the reordered view need not be a strict permutation. This iterator is
//! useful for fusing a scatter or gather operation with other algorithms.
//!
//! This iterator takes two arguments:
//!
//!   - an iterator to the range \c V on which the "permutation" will be applied
//!   - the reindexing scheme that defines how the elements of \c V will be permuted.
//!
//! Note that \p permutation_iterator is not limited to strict permutations of the given range \c V. The distance
//! between begin and end of the reindexing iterators is allowed to be smaller compared to the size of the range \c V,
//! in which case the \p permutation_iterator only provides a "permutation" of a subrange of \c V. The indices neither
//! need to be unique. In this same context, it must be noted that the past-the-end \p permutation_iterator is
//! completely defined by means of the past-the-end iterator to the indices.
//!
//! The following code snippet demonstrates how to create a \p permutation_iterator which represents a reordering of the
//! contents of a \p device_vector.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<float> values{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
//! thrust::device_vector<int> indices{2, 6, 1, 3};
//!
//! using ElementIterator = thrust::device_vector<float>::iterator;
//! using IndexIterator = thrust::device_vector<int>::iterator  ;
//!
//! cuda::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());
//!
//! *iter;   // returns 30.0f;
//! iter[0]; // returns 30.0f;
//! iter[1]; // returns 70.0f;
//! iter[2]; // returns 20.0f;
//! iter[3]; // returns 40.0f;
//!
//! // iter[4] is an out-of-bounds error
//!
//! *iter   = -1.0f; // sets values[2] to -1.0f;
//! iter[0] = -1.0f; // sets values[2] to -1.0f;
//! iter[1] = -1.0f; // sets values[6] to -1.0f;
//! iter[2] = -1.0f; // sets values[1] to -1.0f;
//! iter[3] = -1.0f; // sets values[3] to -1.0f;
//!
//! // values is now {10, -1, -1, -1, 50, 60, -1, 80}
//! @endcode
template <class _Iter, class _Offset = _Iter>
class permutation_iterator
{
private:
  _Iter __iter_     = {};
  _Offset __offset_ = {};

  // We need to factor these out because old gcc chokes with using arguments in friend functions
  template <class _Iter1>
  static constexpr bool __nothrow_difference = noexcept(_CUDA_VSTD::declval<_Iter1>() - _CUDA_VSTD::declval<_Iter1>());

  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_equality = noexcept(_CUDA_VSTD::declval<_Iter1>() == _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_less_than = noexcept(_CUDA_VSTD::declval<_Iter1>() < _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_less_equal = noexcept(_CUDA_VSTD::declval<_Iter1>() <= _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_greater_than =
    noexcept(_CUDA_VSTD::declval<_Iter1>() > _CUDA_VSTD::declval<_Iter2>());
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_greater_equal =
    noexcept(_CUDA_VSTD::declval<_Iter1>() >= _CUDA_VSTD::declval<_Iter2>());

public:
  using iterator_type         = _Iter;
  using iterator_concept      = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category     = _CUDA_VSTD::random_access_iterator_tag;
  using value_type            = _CUDA_VSTD::iter_value_t<_Iter>;
  using difference_type       = _CUDA_VSTD::iter_difference_t<_Iter>;
  using __offset_value_t      = _CUDA_VSTD::iter_value_t<_Offset>;
  using __offset_difference_t = _CUDA_VSTD::iter_difference_t<_Offset>;

  //! Ensure that the user passes an iterator to something interger_like
  static_assert(_CUDA_VSTD::__integer_like<__offset_value_t>,
                "cuda::permutation_iterator: _Offset must be an iterator to integer_like");

  //! Ensure that the offset value_type is convertible to difference_type
  static_assert(_CUDA_VSTD::is_convertible_v<__offset_value_t, difference_type>,
                "cuda::permutation_iterator: _Offsets value type must be convertible to iter_difference<Iter>");

  //! To actually use operator+ we need the offset iterator to be random access
  static_assert(_CUDA_VSTD::random_access_iterator<_Offset>,
                "cuda::permutation_iterator: _Offset must be a random access iterator!");

  //! To actually use operator+ we need the base iterator to be random access
  static_assert(_CUDA_VSTD::random_access_iterator<_Iter>,
                "cuda::permutation_iterator: _Iter must be a random access iterator!");

  //! @brief Default constructs an \p permutation_iterator with a value initialized iterator and offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr permutation_iterator() = default;

  //! @brief Constructs an \p permutation_iterator from an iterator and an optional offset
  //! @param __iter The iterator to to offset from
  //! @param __offset The iterator with the permutations
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator(_Iter __iter, _Offset __offset) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
      : __iter_(__iter)
      , __offset_(__offset)
  {}

  //! @brief Returns a const reference to the stored iterator we are offsetting from
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __iter_;
  }

  //! @brief Extracts the stored iterator we are offsetting from
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Iter>)
  {
    return _CUDA_VSTD::move(__iter_);
  }

  //! @brief Returns a const reference to the offset iterator
  [[nodiscard]] _CCCL_API constexpr const _Offset& offset() const& noexcept
  {
    return __offset_;
  }

  //! @brief Extracts the stored iterator we are offsetting from
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Offset offset() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Offset>)
  {
    return _CUDA_VSTD::move(__offset_);
  }

  //! @brief Dereferences the stored iterator offset by \p *__offset_
  //! @returns __iter_[*__offset_]
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator*() noexcept(noexcept(__iter_[static_cast<difference_type>(*__offset_)]))
  {
    return __iter_[static_cast<difference_type>(*__offset_)];
  }

  //! @brief Dereferences the stored iterator offset by \p *__offset_
  //! @returns __iter_[*__offset_]
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
    noexcept(noexcept(__iter_[static_cast<difference_type>(*__offset_)]))
  {
    return __iter_[static_cast<difference_type>(*__offset_)];
  }

  //! @brief Subscripts the stored iterator by \p __n
  //! @param __n The additional offset
  //! @returns __iter_[__offset_[__n]]
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto)
  operator[](difference_type __n) noexcept(noexcept(__iter_[static_cast<difference_type>(__offset_[__n])]))
  {
    return __iter_[static_cast<difference_type>(__offset_[__n])];
  }

  //! @brief Subscripts the stored iterator by \p __n
  //! @param __n The additional offset
  //! @returns __iter_[__offset_[__n]]
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](difference_type __n) const
    noexcept(noexcept(__iter_[static_cast<difference_type>(__offset_[__n])]))
  {
    return __iter_[static_cast<difference_type>(__offset_[__n])];
  }

  //! @brief Increments the stored offset iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator++() noexcept(noexcept(++__offset_))
  {
    ++__offset_;
    return *this;
  }

  //! @brief Increments the stored offset iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator operator++(int) noexcept(
    noexcept(++__offset_)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
  {
    permutation_iterator __tmp = *this;
    ++__offset_;
    return __tmp;
  }

  //! @brief Decrements the stored offset iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator& operator--() noexcept(noexcept(--__offset_))
  {
    --__offset_;
    return *this;
  }

  //! @brief Decrements the stored offset iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator operator--(int) noexcept(
    noexcept(--__offset_)
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
  {
    permutation_iterator __tmp = *this;
    --__offset_;
    return __tmp;
  }

  //! @brief Advances the stored offset iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr permutation_iterator operator+(difference_type __n) const
    noexcept(noexcept(__offset_ + static_cast<__offset_difference_t>(__n))
             && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter>
             && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
  {
    return permutation_iterator{__iter_, __offset_ + static_cast<__offset_difference_t>(__n)};
  }

  //! @brief Returns a copy of \p __x advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __x The original \c permutation_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr permutation_iterator
  operator+(difference_type __n, const permutation_iterator& __x) noexcept(
    noexcept(__offset_ + static_cast<__offset_difference_t>(__n))
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
  {
    return __x + __n;
  }

  //! @brief Advances the \c permutation_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator&
  operator+=(difference_type __n) noexcept(noexcept(__offset_ += static_cast<__offset_difference_t>(__n)))
  {
    __offset_ += static_cast<__offset_difference_t>(__n);
    return *this;
  }

  //! @brief Returns a copy of the \c permutation_iterator decremented by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr permutation_iterator operator-(difference_type __n) const
    noexcept(noexcept(__offset_ - static_cast<__offset_difference_t>(__n))
             && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter>
             && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
  {
    return permutation_iterator{__iter_, __offset_ - static_cast<__offset_difference_t>(__n)};
  }

  //! @brief Decrements the \c permutation_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr permutation_iterator&
  operator-=(difference_type __n) noexcept(noexcept(__offset_ -= static_cast<__offset_difference_t>(__n)))
  {
    __offset_ -= static_cast<__offset_difference_t>(__n);
    return *this;
  }

  //! @brief Returns the difference in offset between two \c permutation_iterators. Returns distance between offsets
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API friend constexpr difference_type operator-(
    const permutation_iterator& __lhs, const permutation_iterator& __rhs) noexcept(__nothrow_difference<_Offset>)
  {
    return __lhs.__offset_ - __rhs.offset();
  }

  //! @brief Compares two \c permutation_iterator for equality, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_equality<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ == __rhs.offset();
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c permutation_iterator for inequality, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_equality<_Offset, _OtherOffset>)
  {
    return !(__lhs.__offset_ == __rhs.offset());
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  template <class _Iter1, class _Iter2>
  static constexpr bool __nothrow_three_way = noexcept(_CUDA_VSTD::declval<_Iter1>() <=> _CUDA_VSTD::declval<_Iter2>());

  //! @brief Three-way-compares two \c permutation_iterator for inequality, by three-way-comparing the memory location
  //! they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::three_way_comparable_with<_Offset, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering operator<=>(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_three_way<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ <=> __rhs.offset();
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Compares two \c permutation_iterator for less than, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Offset, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_less_than<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ < __rhs.offset();
  }

  //! @brief Compares two \c permutation_iterator for less equal, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Offset, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_less_equal<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ <= __rhs.offset();
  }

  //! @brief Compares two \c permutation_iterator for greater than, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Offset, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator>(
    const permutation_iterator& __lhs,
    const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(__nothrow_greater_than<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ > __rhs.offset();
  }

  //! @brief Compares two \c permutation_iterator for greater equal, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherOffset)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Offset, _OtherOffset>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const permutation_iterator& __lhs, const permutation_iterator<_OtherIter, _OtherOffset>& __rhs) noexcept(
    __nothrow_greater_equal<_Offset, _OtherOffset>)
  {
    return __lhs.__offset_ >= __rhs.offset();
  }
};

_CCCL_TEMPLATE(class _Iter, class _Offset)
_CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter> _CCCL_AND _CUDA_VSTD::random_access_iterator<_Offset>)
_CCCL_HOST_DEVICE permutation_iterator(_Iter, _Offset) -> permutation_iterator<_Iter, _Offset>;

//! @brief Creates an \c permutation_iterator from an iterator and an iterator to an integral offset
//! @param __iter The iterator
//! @param __offset The iterator to an integral offset
_CCCL_TEMPLATE(class _Iter, class _Offset)
_CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter> _CCCL_AND _CUDA_VSTD::random_access_iterator<_Offset>)
[[nodiscard]] _CCCL_API constexpr permutation_iterator<_Iter, _Offset>
make_permutation_iterator(_Iter __iter, _Offset __offset) noexcept(
  _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Offset>)
{
  return permutation_iterator<_Iter, _Offset>{__iter, __offset};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_PERMUTATION_ITERATOR_H
