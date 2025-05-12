//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_OFFSET_ITERATOR_H
#define _CUDA___ITERATOR_OFFSET_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/prev.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/detail/libcxx/include/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _OffsetType, bool = _CUDA_VSTD::__integer_like<_OffsetType>>
struct __offset_iterator_offset_value_type
{
  using type = _OffsetType;
};

template <class _OffsetType>
struct __offset_iterator_offset_value_type<_OffsetType, false>
{
  using type = _CUDA_VSTD::iter_value_t<_OffsetType>;
};

template <class _OffsetType, bool = _CUDA_VSTD::__integer_like<_OffsetType>>
struct __offset_iterator_offset_difference_type
{
  using type = _OffsetType;
};

template <class _OffsetType>
struct __offset_iterator_offset_difference_type<_OffsetType, false>
{
  using type = _CUDA_VSTD::iter_difference_t<_OffsetType>;
};

//! \p offset_iterator wraps another iterator and an integral offset, applies the offset to the iterator when
//! dereferencing, comparing, or computing the distance between two offset_iterators. This is useful, when the
//! underlying iterator cannot be incremented, decremented, or advanced (e.g., because those operations are only
//! supported in device code).
//!
//! The following code snippet demonstrates how to create an \p offset_iterator:
//!
//! \code
//! #include <cuda/iterator>
//! #include <thrust/fill.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> data{1, 2, 3, 4};
//!   auto b = offset_iterator{data.begin(), 1};
//!   auto e = offset_iterator{data.end(), -1};
//!   thrust::fill(thust::device, b, e, 42);
//!   // data is now [1, 42, 42, 4]
//!   ++b; // does not call ++ on the underlying iterator
//!   assert(b == e - 1);
//!
//!   return 0;
//! }
//! \endcode
//!
//! Alternatively, an \p offset_iterator can also use an iterator to retrieve the offset from an iterator. However, such
//! an \p offset_iterator cannot be moved anymore by changing the offset, so it will move the base iterator instead.
//!
//! \code
//! #include <cuda/iterator>
//! #include <thrust/fill.h>
//! #include <thrust/functional.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   using thrust::placeholders::_1;
//!   thrust::device_vector<int> data{1, 2, 3, 4};
//!
//!   thrust::device_vector<ptrdiff> offsets{1}; // offset is only available on device
//!   auto offset = cuda::transform_iterator{offsets.begin(), _1 * 2};
//!   thrust::offset_iterator iter(data.begin(), offset); // load and transform offset upon access
//!   // iter is at position 2 (= 1 * 2) in data, and would return 3 in device code
//!
//!   return 0;
//! }
//! \endcode
//!
//! In the above example, the offset is loaded from a device vector, transformed by a \p transform_iterator, and then
//! applied to the underlying iterator, when the \p offset_iterator is accessed.
template <class _Iter, class _OffsetType = _CUDA_VSTD::iter_difference_t<_Iter>>
class offset_iterator
{
private:
  _Iter __iter_         = {};
  _OffsetType __offset_ = {};

public:
  using iterator_type     = _Iter;
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _CUDA_VSTD::iter_value_t<_Iter>;
  using difference_type   = _CUDA_VSTD::iter_difference_t<_Iter>;

  using __offset_value_t      = typename __offset_iterator_offset_value_type<_OffsetType>::type;
  using __offset_difference_t = typename __offset_iterator_offset_difference_type<_OffsetType>::type;

  //! Ensure that the user passes either something integer_like or an iterator to something interger_like
  static_assert(_CUDA_VSTD::__integer_like<__offset_value_t>,
                "cuda::offset_iterator: _OffsetType must either be interger_like or an iterator to integer_like");

  //! Ensure that the offset type is convertible to difference_type
  static_assert(_CUDA_VSTD::is_convertible_v<__offset_value_t, difference_type>,
                "cuda::offset_iterator: _OffsetType must either be convertible to iter_difference<Iter> or an iterator "
                "to a type convertible to iter_difference<Iter>");

  //! To actually use operator+ we need the base iterator to be random access
  static_assert(_CUDA_VSTD::random_access_iterator<_Iter>,
                "cuda::offset_iterator: _Iter must be a random access iterator!");
  //! To actually use operator+ we need the offset iterator to be random access
  static_assert(_CUDA_VSTD::__integer_like<_OffsetType> || _CUDA_VSTD::random_access_iterator<_OffsetType>,
                "cuda::offset_iterator: _OffsetType must either be an integer type or a random access iterator!");

  //! @brief Default constructs an \p offset_iterator with a value initialized iterator and offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr offset_iterator() = default;

  //! @brief Constructs an \p offset_iterator from an iterator and an optional offset
  //! @param __iter The iterator to to offset from
  //! @param __offset Optional offset. The default is a value initialized _OffsetType
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator(_Iter __iter, _OffsetType __offset = {}) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_OffsetType>)
      : __iter_(__iter)
      , __offset_(__offset)
  {}

  //! @brief Returns a const reference to the stored iterator we are offsetting from
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Iter& base() const& noexcept
  {
    return __iter_;
  }

  //! @brief Extracts the stored iterator we are offsetting from
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Iter base() &&
  {
    return _CUDA_VSTD::move(__iter_);
  }

  //! @brief Returns the current offset we are applying
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr difference_type offset() const noexcept
  {
    if constexpr (_CUDA_VSTD::indirectly_readable<_OffsetType>)
    {
      return static_cast<difference_type>(*__offset_);
    }
    else
    {
      return static_cast<difference_type>(__offset_);
    }
  }

  //! @brief Dereferences the stored iterator offset by \p offset()
  //! @returns *(iter + offset())
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*()
  {
    return *(__iter_ + offset());
  }

  //! @brief Dereferences the stored iterator offset by \p offset()
  //! @returns *(iter + offset())
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _I2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
  {
    return *(__iter_ + offset());
  }

  //! @brief Dereferences the stored iterator offset by \p offset() + \p __n
  //! @param __n The additional offset
  //! @returns *(iter + offset() + n)
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
  {
    return *(__iter_ + offset() + __n);
  }

  //! @brief Increments the stored offset
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator& operator++()
  {
    ++__offset_;
    return *this;
  }

  //! @brief Increments the stored offset
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator operator++(int)
  {
    offset_iterator __tmp = *this;
    ++__offset_;
    return __tmp;
  }

  //! @brief Decrements the stored offset
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator& operator--()
  {
    --__offset_;
    return *this;
  }

  //! @brief Decrements the stored offset
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator operator--(int)
  {
    offset_iterator __tmp = *this;
    --__offset_;
    return __tmp;
  }

  //! @brief Advances the stored offset by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator operator+(difference_type __n) const
  {
    if constexpr (_CUDA_VSTD::__integer_like<_OffsetType>)
    {
      return offset_iterator{__iter_, __offset_ + static_cast<__offset_difference_t>(__n)};
    }
    else
    {
      return offset_iterator{__iter_, _CUDA_VSTD::next(__offset_, static_cast<__offset_difference_t>(__n))};
    }
  }

  //! @brief Returns a copy of \p __x advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __x The original \c offset_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr offset_iterator
  operator+(difference_type __n, const offset_iterator& __x)
  {
    return __x + __n;
  }

  //! @brief Advances the \c offset_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator& operator+=(difference_type __n)
  {
    if constexpr (_CUDA_VSTD::__integer_like<_OffsetType>)
    {
      __offset_ += static_cast<__offset_difference_t>(__n);
    }
    else
    {
      _CUDA_VSTD::advance(__offset_, static_cast<__offset_difference_t>(__n));
    }
    return *this;
  }

  //! @brief Returns a copy of the \c offset_iterator decremented by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator operator-(difference_type __n) const
  {
    if constexpr (_CUDA_VSTD::__integer_like<_OffsetType>)
    {
      return offset_iterator{__iter_, __offset_ - static_cast<__offset_difference_t>(__n)};
    }
    else
    {
      return offset_iterator{__iter_, _CUDA_VSTD::prev(__offset_, static_cast<__offset_difference_t>(__n))};
    }
  }

  //! @brief Returns the difference in offset between two \c offset_iterators
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const offset_iterator& __lhs, const offset_iterator& __rhs)
  {
    return static_cast<difference_type>(__rhs.offset() - __lhs.offset());
  }

  //! @brief Returns the offset between an \c offset_iterators and default sentinel, equivalent to -offset()
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const offset_iterator& __lhs, _CUDA_VSTD::default_sentinel_t)
  {
    return static_cast<difference_type>(-__lhs.offset());
  }

  //! @brief Returns the offset between default sentinel and an \c offset_iterators, equivalent to offset()
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(_CUDA_VSTD::default_sentinel_t, const offset_iterator& __rhs)
  {
    return static_cast<difference_type>(__rhs.offset());
  }

  //! @brief Decrements the \c offset_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator& operator-=(difference_type __n)
  {
    __offset_ -= static_cast<__offset_difference_t>(__n);
    return *this;
  }

  //! @brief Compares two \c offset_iterator for equality, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) == (__rhs.__iter_ + __rhs.offset());
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c offset_iterator for inequality, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) != (__rhs.__iter_ + __rhs.offset());
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares a \c offset_iterator with \c default_sentinel for equality. True if offset() is zero
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const offset_iterator& __lhs, _CUDA_VSTD::default_sentinel_t) noexcept
  {
    return __lhs.offset() == 0;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares a \c offset_iterator with \c default_sentinel for equality. True if offset() is zero
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(_CUDA_VSTD::default_sentinel_t, const offset_iterator& __lhs) noexcept
  {
    return __lhs.offset() == 0;
  }

  //! @brief Compares a \c offset_iterator with \c default_sentinel for inequality. True if offset() is not zero
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const offset_iterator& __lhs, _CUDA_VSTD::default_sentinel_t) noexcept
  {
    return __lhs.offset() != 0;
  }

  //! @brief Compares a \c offset_iterator with \c default_sentinel for inequality. True if offset() is not zero
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(_CUDA_VSTD::default_sentinel_t, const offset_iterator& __lhs) noexcept
  {
    return __lhs.offset() != 0;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c offset_iterator for inequality, by three-way-comparing the memory location
  //! they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr strong_ordering
  operator<=>(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) <=> (__rhs.__iter_ + __rhs.offset());
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Compares two \c offset_iterator for less than, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) < (__rhs.__iter_ + __rhs.offset());
  }

  //! @brief Compares two \c offset_iterator for less equal, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<=(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) <= (__rhs.__iter_ + __rhs.offset());
  }

  //! @brief Compares two \c offset_iterator for greater than, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) > (__rhs.__iter_ + __rhs.offset());
  }

  //! @brief Compares two \c offset_iterator for greater equal, by comparing the memory location they point at
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>=(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept
  {
    return (__lhs.__iter_ + __lhs.offset()) >= (__rhs.__iter_ + __rhs.offset());
  }

  //! @brief Swaps the elements pointed to by \p __lhs and \p __rhs
  //! @param __lhs The left \c offset_iterator
  //! @param __rhs The right \c offset_iterator
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  iter_swap(const offset_iterator& __lhs, const offset_iterator& __rhs) noexcept(
    noexcept(_CUDA_VRANGES::iter_swap(__lhs.__iter_ + __lhs.offset(), __rhs.__iter_ + __rhs.offset())))
  {
    return _CUDA_VRANGES::iter_swap(__lhs.__iter_ + __lhs.offset(), __rhs.__iter_ + __rhs.offset());
  }

  //! @brief Moves the element pointed to by \p __iter
  //! @param __iter The \c offset_iterator
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr decltype(auto) iter_move(
    const offset_iterator& __iter) noexcept(noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<const _Iter&>())))
  {
    return _CUDA_VRANGES::iter_move(__iter.__iter_ + __iter.offset());
  }
};

template <class _Iter>
_CCCL_HOST_DEVICE offset_iterator(_Iter) -> offset_iterator<_Iter, _CUDA_VSTD::iter_difference_t<_Iter>>;

_CCCL_TEMPLATE(class _Iter, class _OffsetType)
_CCCL_REQUIRES(_CUDA_VSTD::__integer_like<_OffsetType>)
_CCCL_HOST_DEVICE offset_iterator(_Iter, _OffsetType) -> offset_iterator<_Iter, _CUDA_VSTD::iter_difference_t<_Iter>>;

//! @brief Creates an \c offset_iterator from an iterator and an integral offset
//! @param __iter The iterator
//! @param __offset The integral offset
_CCCL_TEMPLATE(class _Iter, class _OffsetType)
_CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter> _CCCL_AND _CUDA_VSTD::__integer_like<_OffsetType>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator<_Iter, _CUDA_VSTD::iter_difference_t<_Iter>>
make_offset_iterator(_Iter __iter, _OffsetType __offset)
{
  return offset_iterator<_Iter, _CUDA_VSTD::iter_difference_t<_Iter>>{__iter, __offset};
}

//! @brief Creates an \c offset_iterator from an iterator and an iterator to an integral offset
//! @param __iter The iterator
//! @param __offset The iterator to an integral offset
_CCCL_TEMPLATE(class _Iter, class _OffsetIter)
_CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter> _CCCL_AND _CUDA_VSTD::random_access_iterator<_OffsetIter>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr offset_iterator<_Iter, _OffsetIter>
make_offset_iterator(_Iter __iter, _OffsetIter __offset)
{
  return offset_iterator<_Iter, _OffsetIter>{__iter, __offset};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_OFFSET_ITERATOR_H
