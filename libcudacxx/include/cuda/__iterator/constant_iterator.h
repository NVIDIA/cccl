//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_CONSTANT_ITERATOR_H
#define _CUDA___ITERATOR_CONSTANT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief The \c constant_iterator class represents an iterator in a sequence of repeated values.
template <class _Tp, class _Index = ::cuda::std::ptrdiff_t>
class constant_iterator
{
private:
  static_assert(::cuda::std::__integer_like<_Index>, "The index type of cuda::constant_iterator must be integer-like!");

  ::cuda::std::ranges::__movable_box<_Tp> __value_{::cuda::std::in_place};
  _Index __index_ = 0;

public:
  using iterator_concept  = ::cuda::std::random_access_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = ::cuda::std::ptrdiff_t;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI constant_iterator()
    requires ::cuda::std::default_initializable<_Tp>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Tp2>)
  _CCCL_API constexpr constant_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Tp2>) {}
#endif // !_CCCL_HAS_CONCEPTS()

  //! @brief Creates \c constant_iterator from a \p __value. The index is set to zero
  //! @param __value The value to store in the \c constant_iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator(_Tp __value) noexcept(::cuda::std::is_nothrow_move_constructible_v<_Tp>)
      : __value_(::cuda::std::in_place, ::cuda::std::move(__value))
      , __index_()
  {}

  //! @brief Creates \c constant_iterator from a \p __value and an index
  //! @param __value The value to store in the \c constant_iterator
  //! @param __index The index in the sequence represented by this \c constant_iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(typename _Index2)
  _CCCL_REQUIRES(::cuda::std::__integer_like<_Index2>)
  _CCCL_API constexpr explicit constant_iterator(_Tp __value, _Index2 __index) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Tp>)
      : __value_(::cuda::std::in_place, ::cuda::std::move(__value))
      , __index_(static_cast<_Index>(__index))
  {}

  //! @brief Returns a the current index
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return static_cast<difference_type>(__index_);
  }

  //! @brief Returns a const reference to the stored value
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator*() const noexcept
  {
    return *__value_;
  }

  //! @brief Returns a const reference to the stored value
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](difference_type) const noexcept
  {
    return *__value_;
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr constant_iterator& operator++() noexcept
  {
    ++__index_;
    return *this;
  }

  //! @brief Increments the stored index
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator operator++(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Tp>)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr constant_iterator& operator--() noexcept
  {
    _CCCL_ASSERT(__index_ > 0, "The index must be greater than or equal to 0");
    --__index_;
    return *this;
  }

  //! @brief Decrements the stored index
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr constant_iterator operator--(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Tp>)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Advances the stored index by \p __n
  //! @param __n The amount of elements to advance
  _CCCL_API constexpr constant_iterator& operator+=(difference_type __n) noexcept
  {
    _CCCL_ASSERT(__index_ + __n >= 0, "The index must be greater than or equal to 0");
    __index_ += static_cast<_Index>(__n);
    return *this;
  }

  //! @brief Decrements the stored index by \p __n
  //! @param __n The amount of elements to decrement
  _CCCL_API constexpr constant_iterator& operator-=(difference_type __n) noexcept
  {
    _CCCL_ASSERT(__index_ - __n >= 0, "The index must be greater than or equal to 0");
    __index_ -= static_cast<_Index>(__n);
    return *this;
  }

  //! @brief Compares two \c constant_iterator for equality. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ == __rhs.__index_;
  }
#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c constant_iterator for inequality. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ != __rhs.__index_;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c constant_iterator. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <=> __rhs.__index_;
  }
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR()

  //! @brief Compares two \c constant_iterator for less than. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ < __rhs.__index_;
  }
  //! @brief Compares two \c constant_iterator for less equal. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <= __rhs.__index_;
  }
  //! @brief Compares two \c constant_iterator for greater than. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ > __rhs.__index_;
  }
  //! @brief Compares two \c constant_iterator for greater equal. Only compares the index in the sequence
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return __lhs.__index_ >= __rhs.__index_;
  }

  //! @brief Advances a \c constant_iterator \p __iter  by \p __n
  //! @param __iter The \c constant_iterator to advance
  //! @param __n The amount of elements to advance
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator+(constant_iterator __iter, difference_type __n) noexcept
  {
    __iter += __n;
    return __iter;
  }

  //! @brief Advances a \c constant_iterator \p __iter  by \p __n
  //! @param __n The amount of elements to advance
  //! @param __iter The \c constant_iterator to advance
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator+(difference_type __n, constant_iterator __iter) noexcept
  {
    __iter += __n;
    return __iter;
  }

  //! @brief Decrements a \c constant_iterator \p __iter  by \p __n
  //! @param __iter The \c constant_iterator to decrement
  //! @param __n The amount of elements to decrement
  [[nodiscard]] _CCCL_API friend constexpr constant_iterator
  operator-(constant_iterator __iter, difference_type __n) noexcept
  {
    __iter -= __n;
    return __iter;
  }

  //! @brief Returns the distance between two \c constant_iterator
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const constant_iterator& __lhs, const constant_iterator& __rhs) noexcept
  {
    return static_cast<difference_type>(__lhs.__index_) - static_cast<difference_type>(__rhs.__index_);
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE constant_iterator(_Tp) -> constant_iterator<_Tp, ::cuda::std::ptrdiff_t>;

_CCCL_TEMPLATE(class _Tp, typename _Index)
_CCCL_REQUIRES(::cuda::std::__integer_like<_Index>)
_CCCL_HOST_DEVICE constant_iterator(_Tp, _Index) -> constant_iterator<_Tp, _Index>;

//! @brief make_constant_iterator creates a bounded \p constant_iterator from a value and an index
//! @param __value The value to be returned by the \p constant_iterator
//! @param __index The optional index of the \p constant_iterator representing the position in a sequence. Defaults to 0
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto make_constant_iterator(_Tp __value, ::cuda::std::ptrdiff_t __index = 0)
{
  return constant_iterator<_Tp>{::cuda::std::move(__value), __index};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_CONSTANT_ITERATOR_H
