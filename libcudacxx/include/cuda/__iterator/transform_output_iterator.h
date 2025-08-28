//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H
#define _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H

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
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Iter, class _Fn>
class __transform_output_proxy
{
private:
  template <class, class>
  friend class transform_output_iterator;

  _Iter __iter_;
  _Fn& __func_;

  template <class _MaybeConstFn, class _Arg>
  using _Ret = ::cuda::std::invoke_result_t<_MaybeConstFn&, _Arg>;

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit __transform_output_proxy(_Iter __iter, _Fn& __func) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
      : __iter_(__iter)
      , __func_(__func)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __transform_output_proxy>)
                   _CCCL_AND ::cuda::std::is_invocable_v<_Fn&, _Arg> _CCCL_AND ::cuda::std::
                     is_assignable_v<::cuda::std::iter_reference_t<_Iter>, ::cuda::std::invoke_result_t<_Fn&, _Arg>>)
  _CCCL_API constexpr __transform_output_proxy&
  operator=(_Arg&& __arg) noexcept(noexcept(*__iter_ = ::cuda::std::invoke(__func_, ::cuda::std::forward<_Arg>(__arg))))
  {
    *__iter_ = ::cuda::std::invoke(__func_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(
    (!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __transform_output_proxy>)
      _CCCL_AND ::cuda::std::is_invocable_v<const _Fn&, _Arg> _CCCL_AND ::cuda::std::
        is_assignable_v<::cuda::std::iter_reference_t<const _Iter>, ::cuda::std::invoke_result_t<const _Fn&, _Arg>>)
  _CCCL_API constexpr const __transform_output_proxy& operator=(_Arg&& __arg) const
    noexcept(noexcept(*__iter_ = ::cuda::std::invoke(__func_, ::cuda::std::forward<_Arg>(__arg))))
  {
    *__iter_ = ::cuda::std::invoke(__func_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }
};

//! @brief \p transform_output_iterator is a special kind of output iterator which transforms a value written upon
//! dereference. This iterator is useful for transforming an output from algorithms without explicitly storing the
//! intermediate result in the memory and applying subsequent transformation, thereby avoiding wasting memory capacity
//! and bandwidth. Using \p transform_iterator facilitates kernel fusion by deferring execution of transformation until
//! the value is written while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrated how to create a \p transform_output_iterator which applies \c sqrtf to the
//! assigning value.
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
//!     return cuda::std::sqrtf(x);
//!   }
//! };
//!
//! int main()
//! {
//!   thrust::device_vector<float> v(4);
//!   cuda::transform_output_iterator iter(v.begin(), square_root());
//!
//!   iter[0] =  1.0f;    // stores sqrtf( 1.0f)
//!   iter[1] =  4.0f;    // stores sqrtf( 4.0f)
//!   iter[2] =  9.0f;    // stores sqrtf( 9.0f)
//!   iter[3] = 16.0f;    // stores sqrtf(16.0f)
//!   // iter[4] is an out-of-bounds error
//!
//!   v[0]; // returns 1.0f;
//!   v[1]; // returns 2.0f;
//!   v[2]; // returns 3.0f;
//!   v[3]; // returns 4.0f;
//!
//! }
//! @endcode
template <class _Iter, class _Fn>
class transform_output_iterator
{
  static_assert(::cuda::std::is_object_v<_Fn>,
                "cuda::transform_output_iterator requires that _Fn is a function object");

public:
  _Iter __current_{};
  ::cuda::std::ranges::__movable_box<_Fn> __func_{};

  using iterator_concept  = ::cuda::std::output_iterator_tag;
  using iterator_category = ::cuda::std::output_iterator_tag;
  using difference_type   = ::cuda::std::iter_difference_t<_Iter>;
  using value_type        = void;
  using pointer           = void;
  using reference         = void;

  //! @brief Default constructs a \p transform_output_iterator with a value initialized iterator and functor
#if _CCCL_HAS_CONCEPTS()
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI transform_output_iterator()
    requires ::cuda::std::default_initializable<_Iter> && ::cuda::std::default_initializable<_Fn>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _Fn2 = _Fn)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Iter2> _CCCL_AND ::cuda::std::default_initializable<_Fn2>)
  _CCCL_API constexpr transform_output_iterator() noexcept(
    ::cuda::std::is_nothrow_default_constructible_v<_Iter2> && ::cuda::std::is_nothrow_default_constructible_v<_Fn2>)
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! @brief Constructs a \p transform_output_iterator with a given \p __iter iterator and \p __func functor
  //! @param __iter The iterator to transform
  //! @param __func The functor to apply to the iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_output_iterator(_Iter __current, _Fn __func_) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Iter> && ::cuda::std::is_nothrow_move_constructible_v<_Fn>)
      : __current_(::cuda::std::move(__current))
      , __func_(::cuda::std::in_place, ::cuda::std::move(__func_))
  {}

  //! @brief Returns a const reference to the iterator stored in this \p transform_output_iterator
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }

  //! @brief Extracts the iterator stored in this \p transform_output_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() && noexcept(::cuda::std::is_nothrow_move_constructible_v<_Iter>)
  {
    return ::cuda::std::move(__current_);
  }

  //! @brief Returns a proxy that transforms the input upon assignment
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() const noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_output_proxy{__current_, const_cast<_Fn&>(*__func_)};
  }

  //! @brief Returns a proxy that transforms the input upon assignment
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_output_proxy{__current_, *__func_};
  }

  //! @brief Returns a proxy that transforms the input upon assignment storing the current iterator advanced by \p __n
  //! @param __n The additional offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_subscript<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2> && noexcept(__current_ + __n))
  {
    return __transform_output_proxy{__current_ + __n, const_cast<_Fn&>(*__func_)};
  }

  //! @brief Returns a proxy that transforms the input upon assignment storing the current iterator advanced by \p __n
  //! @param __n The additional offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_subscript<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter2> && noexcept(__current_ + __n))
  {
    return __transform_output_proxy{__current_ + __n, const_cast<_Fn&>(*__func_)};
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_output_iterator& operator++() noexcept(noexcept(++__current_))
  {
    ++__current_;
    return *this;
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr auto operator++(int) noexcept(noexcept(++__current_))
  {
    if constexpr (::cuda::std::forward_iterator<_Iter> || ::cuda::std::output_iterator<_Iter, value_type>)
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
  _CCCL_REQUIRES(::cuda::std::__iter_can_decrement<_Iter2>)
  _CCCL_API constexpr transform_output_iterator& operator--() noexcept(noexcept(--__current_))
  {
    --__current_;
    return *this;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_decrement<_Iter2>)
  _CCCL_API constexpr transform_output_iterator
  operator--(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(--__current_))
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Advances this \c transform_output_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_plus_equal<_Iter2>)
  _CCCL_API constexpr transform_output_iterator& operator+=(difference_type __n) noexcept(noexcept(__current_ += __n))
  {
    __current_ += __n;
    return *this;
  }

  //! @brief Decrements this \c transform_output_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::__iter_can_minus_equal<_Iter2>)
  _CCCL_API constexpr transform_output_iterator& operator-=(difference_type __n) noexcept(noexcept(__current_ -= __n))
  {
    __current_ -= __n;
    return *this;
  }

  //! @brief Compares two \c transform_output_iterator for equality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() == ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ == __rhs.__current_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c transform_output_iterator for inequality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator!=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() != ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ != __rhs.__current_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares two \c transform_output_iterator for less than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ < __rhs.__current_;
  }

  //! @brief Compares two \c transform_output_iterator for greater than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ > __rhs.__current_;
  }

  //! @brief Compares two \c transform_output_iterator for less equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ <= __rhs.__current_;
  }

  //! @brief Compares two \c transform_output_iterator for greater equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ >= __rhs.__current_;
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c transform_output_iterator, directly three-way-comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() <=> ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(
      ::cuda::std::random_access_iterator<_Iter2>&& ::cuda::std::three_way_comparable<_Iter2>)
  {
    return __lhs.__current_ <=> __rhs.__current_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  //! @brief Returns a copy of the \c transform_output_iterator \p __i advanced by \p __n
  //! @param __i The \c transform_output_iterator to advance
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(const transform_output_iterator& __i,
            difference_type __n) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
                                          && noexcept(::cuda::std::declval<const _Iter2&>() + difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_output_iterator)(::cuda::std::__iter_can_plus<_Iter2>)
  {
    return transform_output_iterator{__i.__current_ + __n, *__i.__func_};
  }

  //! @brief Returns a copy of the \c transform_output_iterator \p __i advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __i The \c transform_output_iterator to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(difference_type __n, const transform_output_iterator& __i) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
    && noexcept(::cuda::std::declval<const _Iter2&>() + difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_output_iterator)(::cuda::std::__iter_can_plus<_Iter2>)
  {
    return transform_output_iterator{__i.__current_ + __n, *__i.__func_};
  }

  //! @brief Returns a copy of the the \c transform_output_iterator \p __i decremented by \p __n
  //! @param __i The \c transform_output_iterator to decrement
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_output_iterator& __i,
            difference_type __n) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2>
                                          && noexcept(::cuda::std::declval<const _Iter2&>() - difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_output_iterator)(::cuda::std::__iter_can_minus<_Iter2>)
  {
    return transform_output_iterator{__i.__current_ - __n, *__i.__func_};
  }

  //! @brief Returns the distance between \p __lhs and \p __rhs
  //! @param __lhs The left \c transform_output_iterator
  //! @param __rhs The right \c transform_output_iterator
  //! @return The distance between the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_output_iterator& __lhs, const transform_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() - ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(difference_type)(::cuda::std::sized_sentinel_for<_Iter2, _Iter2>)
  {
    return __lhs.__current_ - __rhs.__current_;
  }
};

//! @brief make_transform_output_iterator creates a \p transform_output_iterator from an \c _Iter and a \c _Fn.
//!
//! @param __iter The \c Iterator pointing to the input range of the newly created \p transform_output_iterator.
//! @param __fun The \c _Fn used to transform the range pointed to by @param __iter in the newly created
//! @p transform_output_iterator.
//! @return A new \p transform_output_iterator which transforms the range at @param __iter by @param __fun.
template <class _Iter, class _Fn>
[[nodiscard]] _CCCL_API constexpr auto make_transform_output_iterator(_Iter __iter, _Fn __fun)
{
  return transform_output_iterator<_Iter, _Fn>{__iter, __fun};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TRANSFORM_OUTPUT_ITERATOR_H
