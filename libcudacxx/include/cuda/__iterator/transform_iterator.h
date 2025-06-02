// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_TRANSFORM_ITERATOR_H
#define _CUDA___ITERATOR_TRANSFORM_ITERATOR_H

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
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class, class, class = void>
struct __transform_iterator_category_base
{};

template <class _Iter, class _Fn>
struct __transform_iterator_category_base<_Iter, _Fn, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::forward_iterator<_Iter>>>
{
  using _Cat = typename _CUDA_VSTD::iterator_traits<_Iter>::iterator_category;

  using iterator_category = _CUDA_VSTD::conditional_t<
    _CUDA_VSTD::is_reference_v<_CUDA_VSTD::invoke_result_t<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>>,
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::derived_from<_Cat, _CUDA_VSTD::contiguous_iterator_tag>,
                              _CUDA_VSTD::random_access_iterator_tag,
                              _Cat>,
    _CUDA_VSTD::input_iterator_tag>;
};

template <class _Fn, class _Iter, bool = _CUDA_VSTD::random_access_iterator<_Iter>>
inline constexpr bool __transform_iterator_nothrow_subscript = false;

template <class _Fn, class _Iter>
inline constexpr bool __transform_iterator_nothrow_subscript<_Fn, _Iter, true> =
  noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::declval<_Fn&>(), _CUDA_VSTD::declval<_Iter&>()[0]));

//! \addtogroup iterators
//! \{

//!! \addtogroup fancyiterator Fancy Iterators
//!  \ingroup iterators
//!  \{

//! @brief \p transform_iterator is an iterator which represents a pointer into a range of values after transformation
//! by a function. This iterator is useful for creating a range filled with the result of applying an operation to
//! another range without either explicitly storing it in memory, or explicitly executing the transformation. Using
//! \p transform_iterator facilitates kernel fusion by deferring the execution of a transformation until the value is
//! needed while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a \p transform_iterator which represents the result of
//! \c sqrtf applied to the contents of a \p thrust::device_vector.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! struct square_root
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return sqrtf(x);
//!   }
//! };
//!
//! int main()
//! {
//!   thrust::device_vector<float> v{1.0f, 4.0f, 9.0f, 16.0f};
//!
//!   using FloatIterator = thrust::device_vector<float>::iterator;
//!
//!   cuda::transform_iterator iter(v.begin(), square_root{});
//!
//!   *iter;   // returns 1.0f
//!   iter[0]; // returns 1.0f;
//!   iter[1]; // returns 2.0f;
//!   iter[2]; // returns 3.0f;
//!   iter[3]; // returns 4.0f;
//!
//!   // iter[4] is an out-of-bounds error
//! }
//! @endcode
//!
//! This next example demonstrates how to use a \p transform_iterator with the \p thrust::reduce function to compute the
//! sum of squares of a sequence. We will create temporary \p transform_iterators utilising class template argument
//! deduction avoid explicitly specifying their type:
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//! #include <thrust/reduce.h>
//! #include <iostream>
//!
//! struct square
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return x * x;
//!   }
//! };
//!
//! int main()
//! {
//!   // initialize a device array
//!   thrust::device_vector<float> v(4);
//!   v[0] = 1.0f;
//!   v[1] = 2.0f;
//!   v[2] = 3.0f;
//!   thrust::device_vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
//!   thrust::reduce(cuda::transform_iterator{v.begin(), square{}},
//!                  cuda::transform_iterator{v.end(),   square{}});
//!
//!   std::cout << "sum of squares: " << sum_of_squares << std::endl;
//!   return 0;
//! }
//! @endcode
template <class _Iter, class _Fn>
class transform_iterator : public __transform_iterator_category_base<_Iter, _Fn>
{
  static_assert(_CUDA_VSTD::is_object_v<_Fn>, "cuda::transform_iterator requires that _Fn is a function object");
  static_assert(_CUDA_VSTD::regular_invocable<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>,
                "cuda::transform_iterator requires that _Fn is invocable with iter_reference_t<_Iter>");
  static_assert(_CUDA_VSTD::__can_reference<_CUDA_VSTD::invoke_result_t<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>>,
                "cuda::transform_iterator requires that the return type of _Fn is referenceable");

public:
  _CCCL_NO_UNIQUE_ADDRESS _Iter __current_;
  _CCCL_NO_UNIQUE_ADDRESS _CUDA_VRANGES::__movable_box<_Fn> __func_;

  using iterator_concept = _CUDA_VSTD::conditional_t<
    _CUDA_VSTD::random_access_iterator<_Iter>,
    _CUDA_VSTD::random_access_iterator_tag,
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::bidirectional_iterator<_Iter>,
                              _CUDA_VSTD::bidirectional_iterator_tag,
                              _CUDA_VSTD::conditional_t<_CUDA_VSTD::forward_iterator<_Iter>,
                                                        _CUDA_VSTD::forward_iterator_tag,
                                                        _CUDA_VSTD::input_iterator_tag>>>;
  using value_type = _CUDA_VSTD::remove_cvref_t<_CUDA_VSTD::invoke_result_t<_Fn&, _CUDA_VSTD::iter_reference_t<_Iter>>>;
  using difference_type = _CUDA_VSTD::iter_difference_t<_Iter>;

  //! @brief Default constructs a \p transform_iterator with a value initialized iterator and functor
#if _CCCL_HAS_CONCEPTS()
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI transform_iterator()
    requires _CUDA_VSTD::default_initializable<_Iter> && _CUDA_VSTD::default_initializable<_Fn>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _Fn2 = _Fn)
  _CCCL_REQUIRES(_CUDA_VSTD::default_initializable<_Iter2> _CCCL_AND _CUDA_VSTD::default_initializable<_Fn2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator() noexcept(
    _CUDA_VSTD::is_nothrow_default_constructible_v<_Iter2> && _CUDA_VSTD::is_nothrow_default_constructible_v<_Fn2>)
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! @brief Constructs a \p transform_iterator with a given \p __iter iterator and \p __func functor
  //! @param __iter The iterator to transform
  //! @param __func The functor to apply to the iterator
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator(_Iter __current, _Fn __func_) noexcept(
    _CUDA_VSTD::is_nothrow_move_constructible_v<_Iter> && _CUDA_VSTD::is_nothrow_move_constructible_v<_Fn>)
      : __current_(_CUDA_VSTD::move(__current))
      , __func_(_CUDA_VSTD::in_place, _CUDA_VSTD::move(__func_))
  {}

  //! @brief Returns a const reference to the iterator stored in this \p transform_iterator
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }

  //! @brief Extracts the iterator stored in this \p transform_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Iter
  base() && noexcept(_CUDA_VSTD::is_nothrow_move_constructible_v<_Iter>)
  {
    return _CUDA_VSTD::move(__current_);
  }

  //! @brief Invokes the stored functor with the value pointed to by the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
    noexcept(noexcept(_CUDA_VSTD::invoke(*__func_, *__current_)))
  {
    return _CUDA_VSTD::invoke(*__func_, *__current_);
  }

  //! @brief Invokes the stored functor with the value pointed to by the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  operator*() noexcept(noexcept(_CUDA_VSTD::invoke(*__func_, *__current_)))
  {
    return _CUDA_VSTD::invoke(*__func_, *__current_);
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator& operator++() noexcept(noexcept(++__current_))
  {
    ++__current_;
    return *this;
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator++(int) noexcept(noexcept(++__current_))
  {
    if constexpr (_CUDA_VSTD::forward_iterator<_Iter>)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }
    else
    {
      ++__current_;
    }
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::bidirectional_iterator<_Iter2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator& operator--() noexcept(noexcept(--__current_))
  {
    --__current_;
    return *this;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::bidirectional_iterator<_Iter2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator
  operator--(int) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Iter> && noexcept(--__current_))
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Advances this \c transform_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator&
  operator+=(difference_type __n) noexcept(noexcept(__current_ += __n))
  {
    __current_ += __n;
    return *this;
  }

  //! @brief Decrements this \c transform_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr transform_iterator&
  operator-=(difference_type __n) noexcept(noexcept(__current_ -= __n))
  {
    __current_ -= __n;
    return *this;
  }

  //! @brief Subscripts the stored iterator by \p __n and applies the stored functor to the result
  //! @param __n The additional offset
  //! @returns _CUDA_VSTD::invoke(__func_, __current_[__n])
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter2> _CCCL_AND
                   _CUDA_VSTD::regular_invocable<const _Fn&, _CUDA_VSTD::iter_reference_t<const _Iter2>>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
    noexcept(__transform_iterator_nothrow_subscript<const _Fn, _Iter2>)
  {
    return _CUDA_VSTD::invoke(*__func_, __current_[__n]);
  }

  //! @brief Subscripts the stored iterator by \p __n and applies the stored functor to the result
  //! @param __n The additional offset
  //! @returns _CUDA_VSTD::invoke(__func_, __current_[__n])
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::random_access_iterator<_Iter2>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  operator[](difference_type __n) noexcept(__transform_iterator_nothrow_subscript<_Fn, _Iter2>)
  {
    return _CUDA_VSTD::invoke(*__func_, __current_[__n]);
  }

  //! @brief Compares two \c transform_iterator for equality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator==(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() == _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ == __rhs.__current_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c transform_iterator for inequality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator!=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() != _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ != __rhs.__current_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares two \c transform_iterator for less than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ < __rhs.__current_;
  }

  //! @brief Compares two \c transform_iterator for greater than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator>(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ > __rhs.__current_;
  }

  //! @brief Compares two \c transform_iterator for less equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ <= __rhs.__current_;
  }

  //! @brief Compares two \c transform_iterator for greater equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator>=(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() < _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ >= __rhs.__current_;
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c transform_iterator, directly three-way-comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const transform_iterator& __lhs, const transform_iterator& __rhs) noexcept(
    noexcept(_CUDA_VSTD::declval<const _Iter2&>() <=> _CUDA_VSTD::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VSTD::random_access_iterator<_Iter2>&& _CUDA_VSTD::three_way_comparable<_Iter2>)
  {
    return __lhs.__current_ <=> __rhs.__current_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  //! @brief Returns a copy of the \c transform_iterator \p __i advanced by \p __n
  //! @param __i The \c transform_iterator to advance
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator+(const transform_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return transform_iterator{__i.__current_ + __n, *__i.__func_};
  }

  //! @brief Returns a copy of the \c transform_iterator \p __i advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __i The \c transform_iterator to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator+(difference_type __n, const transform_iterator& __i)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return transform_iterator{__i.__current_ + __n, *__i.__func_};
  }

  //! @brief Returns a copy of the the \c transform_iterator \p __i decremented by \p __n
  //! @param __i The \c transform_iterator to decrement
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator-(const transform_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(transform_iterator)(_CUDA_VSTD::random_access_iterator<_Iter2>)
  {
    return transform_iterator{__i.__current_ - __n, *__i.__func_};
  }

  //! @brief Returns the distance between \p __lhs and \p __rhs
  //! @param __lhs The left \c transform_iterator
  //! @param __rhs The right \c transform_iterator
  //! @return The distance between the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator-(const transform_iterator& __lhs, const transform_iterator& __rhs)
    _CCCL_TRAILING_REQUIRES(difference_type)(_CUDA_VSTD::sized_sentinel_for<_Iter2, _Iter2>)
  {
    return __lhs.__current_ - __rhs.__current_;
  }
};

//! @brief make_transform_iterator creates a \p transform_iterator from an \c _Iter and a \c _Fn.
//!
//! @param __iter The \c Iterator pointing to the input range of the newly created \p transform_iterator.
//! @param __fun The \c _Fn used to transform the range pointed to by @param __iter in the newly created
//! \p transform_iterator.
//! @return A new \p transform_iterator which transforms the range at @param __iter by @param __fun.
template <class _Iter, class _Fn>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto make_transform_iterator(_Iter __iter, _Fn __fun)
{
  return transform_iterator<_Iter, _Fn>{__iter, __fun};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TRANSFORM_ITERATOR_H
