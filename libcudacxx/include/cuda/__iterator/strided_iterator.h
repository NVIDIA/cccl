// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_STRIDED_ITERATOR_H
#define _CUDA___ITERATOR_STRIDED_ITERATOR_H

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
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__mdspan/submdspan_helper.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! A \p strided_iterator wraps another iterator and advances it by a specified stride each time it is incremented or
//! decremented.
//!
//! \param _Iter A random access iterator
//! \param _Stride Either an \ref __integer-like__ or a \ref __integral-constant-like__ specifying the stride
template <class _Iter, class _Stride = _CUDA_VSTD::iter_difference_t<_Iter>>
class strided_iterator
{
private:
  static_assert(_CUDA_VSTD::random_access_iterator<_Iter>,
                "The iterator underlying a strided_iterator must be a random access iterator.");
  static_assert(_CUDA_VSTD::__integer_like<_Stride> || _CUDA_VSTD::__integral_constant_like<_Stride>,
                "The stride of a strided_iterator must either be an integer-like or integral-constant-like.");

  _CCCL_NO_UNIQUE_ADDRESS _Iter __iter_{};
  _CCCL_NO_UNIQUE_ADDRESS _Stride __stride_{};

  template <class, class>
  friend class strided_iterator;

public:
  using iterator_concept  = _CUDA_VSTD::random_access_iterator_tag;
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _CUDA_VSTD::iter_value_t<_Iter>;
  using difference_type   = _CUDA_VSTD::iter_difference_t<_Iter>;

  //! NOTE: _Iter must be default initializable because it is a random_access_iterator and thereby semiregular
  //!       _Stride must be integer-like or integral_constant_like which requires default constructability
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI strided_iterator() = default;

  // We want to avoid constructing a strided_iterator with a value constructed __integer like__ stride, because that
  // would value construct to 0 and incrementing the iterator would do nothing.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Stride2 = _Stride)
  _CCCL_REQUIRES((!_CUDA_VSTD::__integer_like<_Stride2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit strided_iterator(_Iter __iter) noexcept(
    _CUDA_VSTD::is_nothrow_move_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_default_constructible_v<_Stride2>)
      : __iter_(_CUDA_VSTD::move(__iter))
      , __stride_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit strided_iterator(_Iter __iter, _Stride __stride) noexcept(
    _CUDA_VSTD::is_nothrow_move_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_move_constructible_v<_Stride>)
      : __iter_(_CUDA_VSTD::move(__iter))
      , __stride_(_CUDA_VSTD::move(__stride))
  {}

  //! @brief Returns a const reference to the iterator stored in this \p transform_iterator
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Iter& base() const& noexcept
  {
    return __iter_;
  }

  //! @brief Extracts the iterator stored in this \p transform_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Iter
  base() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Iter>)
  {
    return _CUDA_VSTD::move(__iter_);
  }

  static constexpr bool __noexcept_stride =
    noexcept(static_cast<difference_type>(_CUDA_VSTD::__de_ice(_CUDA_VSTD::declval<const _Stride&>())));

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr difference_type stride() const noexcept(__noexcept_stride)
  {
    return static_cast<difference_type>(_CUDA_VSTD::__de_ice(__stride_));
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() noexcept(noexcept(*__iter_))
  {
    return *__iter_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const noexcept(noexcept(*__iter_))
  {
    return *__iter_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  operator[](difference_type __n) noexcept(__noexcept_stride && noexcept(__iter_[__n]))
  {
    return __iter_[__n * stride()];
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__dereferenceable<const _Iter2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
    noexcept(__noexcept_stride && noexcept(__iter_[__n]))
  {
    return __iter_[__n * stride()];
  }

  // Note: we cannot use __iter_ += stride() in the noexcept clause because that breaks gcc < 9
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr strided_iterator&
  operator++() noexcept(__noexcept_stride && noexcept(__iter_ += 1))
  {
    __iter_ += stride();
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator++(int) noexcept(
    noexcept(__noexcept_stride && noexcept(__iter_ += 1))
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Stride>)
  {
    auto __tmp = *this;
    __iter_ += stride();
    return __tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr strided_iterator&
  operator--() noexcept(__noexcept_stride && noexcept(__iter_ -= 1))
  {
    __iter_ -= stride();
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr strided_iterator operator--(int) noexcept(
    noexcept(__noexcept_stride && noexcept(__iter_ -= 1))
    && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_copy_constructible_v<_Stride>)
  {
    auto __tmp = *this;
    __iter_ -= stride();
    return __tmp;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr strided_iterator&
  operator+=(difference_type __n) noexcept(__noexcept_stride && noexcept(__iter_ += 1))
  {
    __iter_ += stride() * __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr strided_iterator&
  operator-=(difference_type __n) noexcept(__noexcept_stride && noexcept(__iter_ -= 1))
  {
    __iter_ -= stride() * __n;
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() == _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter_ == __y.base();
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() == _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter_ != __y.base();
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter_ < __y.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __y < __x;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator<=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return !(__y < __x);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator>=(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() < _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return !(__x < __y);
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::totally_ordered_with<_Iter, _OtherIter>)
  _CCCL_REQUIRES(
    _CUDA_VSTD::totally_ordered<_Iter, _OtherIter> _CCCL_AND _CUDA_VSTD::three_way_comparable_with<_Iter, _OtherIter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter&>() <=> _CUDA_VSTD::declval<const _OtherIter&>()))
  {
    return __x.__iter_ <=> __y.base();
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr strided_iterator
  operator+(strided_iterator __i, difference_type __n) noexcept(noexcept(__iter_ += __n))
  {
    __i += __n;
    return __i;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr strided_iterator
  operator+(difference_type __n, strided_iterator __i) noexcept(noexcept(__iter_ + __n))
  {
    return __i + __n;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr strided_iterator
  operator-(strided_iterator __i, difference_type __n) noexcept(noexcept(__iter_ -= __n))
  {
    __i -= __n;
    return __i;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OtherIter, class _OtherStride)
  _CCCL_REQUIRES(_CUDA_VSTD::sized_sentinel_for<_OtherIter, _Iter>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const strided_iterator& __x, const strided_iterator<_OtherIter, _OtherStride>& __y) noexcept(
    noexcept(_CUDA_VSTD::declval<_Iter>() - _CUDA_VSTD::declval<_OtherIter>()))
  {
    const difference_type __diff = __x.__iter_ - __y.base();
    _CCCL_ASSERT(__x.stride() == __y.stride(), "Taking the difference of two strided_iterators with different stride");
    _CCCL_ASSERT(__diff % __x.stride() == 0, "Underlying iterator difference must be divisible by the stride");
    return __diff / __x.stride();
  }
};

template <class _Iter, typename _Stride>
_CCCL_HOST_DEVICE strided_iterator(_Iter, _Stride) -> strided_iterator<_Iter, _Stride>;

//! @brief make_strided_iterator creates a \p strided_iterator from a random access iterator and an optional stride
//! @param __iter The random_access iterator
//! @param __stride The optional stride. Is value initialized if not provided
template <class _Iter, class _Stride = _CUDA_VSTD::iter_difference_t<_Iter>>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto make_strided_iterator(_Iter __iter, _Stride __stride = {})
{
  return strided_iterator<_Iter, _Stride>{_CUDA_VSTD::move(__iter), _CUDA_VSTD::move(__stride)};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_STRIDED_ITERATOR_H
