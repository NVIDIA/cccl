//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_TRANSFORM_INPUT_OUTPUT_ITERATOR_H
#define _CUDA___ITERATOR_TRANSFORM_INPUT_OUTPUT_ITERATOR_H

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

template <class _Iter, class _InputFn, class _OutputFn>
class __transform_input_output_proxy
{
private:
  template <class, class, class>
  friend class transform_input_output_iterator;

  _Iter __iter_;
  _InputFn& __input_func_;
  _OutputFn& __output_func_;

  using _InputValueType = ::cuda::std::invoke_result_t<_InputFn, ::cuda::std::iter_value_t<_Iter>>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr explicit __transform_input_output_proxy(
    _Iter __iter,
    _InputFn& __input_func,
    _OutputFn& __output_func) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
      : __iter_(__iter)
      , __input_func_(__input_func)
      , __output_func_(__output_func)
  {}

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(
    (!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __transform_input_output_proxy>)
      _CCCL_AND ::cuda::std::is_invocable_v<_OutputFn&, _Arg> _CCCL_AND ::cuda::std::
        is_assignable_v<::cuda::std::iter_reference_t<_Iter>, ::cuda::std::invoke_result_t<_OutputFn&, _Arg>>)
  _CCCL_API constexpr __transform_input_output_proxy& operator=(_Arg&& __arg) noexcept(
    noexcept(*__iter_ = ::cuda::std::invoke(__output_func_, ::cuda::std::forward<_Arg>(__arg))))
  {
    *__iter_ = ::cuda::std::invoke(__output_func_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __transform_input_output_proxy>)
                   _CCCL_AND ::cuda::std::is_invocable_v<const _OutputFn&, _Arg>
                     _CCCL_AND ::cuda::std::is_assignable_v<::cuda::std::iter_reference_t<const _Iter>,
                                                            ::cuda::std::invoke_result_t<const _OutputFn&, _Arg>>)
  _CCCL_API constexpr const __transform_input_output_proxy& operator=(_Arg&& __arg) const
    noexcept(noexcept(*__iter_ = ::cuda::std::invoke(__output_func_, ::cuda::std::forward<_Arg>(__arg))))
  {
    *__iter_ = ::cuda::std::invoke(__output_func_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr operator _InputValueType() const noexcept(noexcept(::cuda::std::invoke(__input_func_, *__iter_)))
  {
    return ::cuda::std::invoke(__input_func_, *__iter_);
  }
};

//! @addtogroup iterators
//! @{

//! @brief \p transform_input_output_iterator is a special kind of iterator which applies transform functions when
//! reading from or writing to dereferenced values. This iterator is useful for algorithms that operate on a type that
//! needs to be serialized/deserialized from values in another iterator, avoiding the need to materialize intermediate
//! results in memory. This also enables the transform functions to be fused with the operations that read and write to
//! the `transform_input_output_iterator`.
//!
//! The following code snippet demonstrates how to create a \p transform_input_output_iterator which performs different
//! transformations when reading from and writing to the iterator.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//!  int main()
//!  {
//!    const size_t size = 4;
//!    thrust::device_vector<float> v(size);
//!
//!    // Write 1.0f, 2.0f, 3.0f, 4.0f to vector
//!    thrust::sequence(v.begin(), v.end(), 1);
//!
//!    // Iterator that negates read values and writes squared values
//!    auto iter = cuda::make_transform_input_output_iterator(v.begin(),
//!        ::cuda::std::negate<float>{}, thrust::square<float>{});
//!
//!    // Iterator negates values when reading
//!    std::cout << iter[0] << " ";  // -1.0f;
//!    std::cout << iter[1] << " ";  // -2.0f;
//!    std::cout << iter[2] << " ";  // -3.0f;
//!    std::cout << iter[3] << "\n"; // -4.0f;
//!
//!    // Write 1.0f, 2.0f, 3.0f, 4.0f to iterator
//!    thrust::sequence(iter, iter + size, 1);
//!
//!    // Values were squared before writing to vector
//!    std::cout << v[0] << " ";  // 1.0f;
//!    std::cout << v[1] << " ";  // 4.0f;
//!    std::cout << v[2] << " ";  // 9.0f;
//!    std::cout << v[3] << "\n"; // 16.0f;
//!
//!  }
//! @endcode
template <class _Iter, class _InputFn, class _OutputFn>
class transform_input_output_iterator
{
public:
  _Iter __current_{};
  ::cuda::std::ranges::__movable_box<_InputFn> __input_func_{};
  ::cuda::std::ranges::__movable_box<_OutputFn> __output_func_{};

  using iterator_concept = ::cuda::std::conditional_t<
    ::cuda::std::random_access_iterator<_Iter>,
    ::cuda::std::random_access_iterator_tag,
    ::cuda::std::conditional_t<::cuda::std::bidirectional_iterator<_Iter>,
                               ::cuda::std::bidirectional_iterator_tag,
                               ::cuda::std::conditional_t<::cuda::std::forward_iterator<_Iter>,
                                                          ::cuda::std::forward_iterator_tag,
                                                          ::cuda::std::output_iterator_tag>>>;
  using iterator_category = ::cuda::std::output_iterator_tag;
  using difference_type   = ::cuda::std::iter_difference_t<_Iter>;
  using value_type        = ::cuda::std::invoke_result_t<_InputFn&, ::cuda::std::iter_reference_t<_Iter>>;
  using pointer           = void;
  using reference         = __transform_input_output_proxy<_Iter, _InputFn, _OutputFn>;

  static_assert(::cuda::std::is_object_v<_InputFn>,
                "cuda::transform_input_output_iterator requires that _InputFn is a function object");
  static_assert(::cuda::std::is_object_v<_OutputFn>,
                "cuda::transform_input_output_iterator requires that _OutputFn is a function object");
  static_assert(::cuda::std::forward_iterator<_Iter> || ::cuda::std::output_iterator<_Iter, value_type>,
                "cuda::transform_input_output_iterator requires that _Iter models forward_iterator or output_iterator");
  static_assert(::cuda::std::is_invocable_v<_InputFn&, ::cuda::std::iter_reference_t<_Iter>>,
                "cuda::transform_input_output_iterator requires that _InputFn is invocable on the result of "
                "dereferencing _Iter");

  //! @brief Default constructs a \p transform_input_output_iterator with a value initialized iterator and functors
#if _CCCL_HAS_CONCEPTS()
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI transform_input_output_iterator()
    requires ::cuda::std::default_initializable<_Iter> && ::cuda::std::default_initializable<_InputFn>
            && ::cuda::std::default_initializable<_OutputFn>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter, class _InputFn2 = _InputFn, class _OutputFn2 = _OutputFn)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Iter2> _CCCL_AND ::cuda::std::default_initializable<_InputFn2>
                   _CCCL_AND ::cuda::std::default_initializable<_OutputFn2>)
  _CCCL_API constexpr transform_input_output_iterator() noexcept(
    ::cuda::std::is_nothrow_default_constructible_v<_Iter2>
    && ::cuda::std::is_nothrow_default_constructible_v<_InputFn2>
    && ::cuda::std::is_nothrow_default_constructible_v<_OutputFn2>)
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! @brief Constructs a \p transform_input_output_iterator with a given \p __iter iterator, \p __input_func and
  //! \p __output_func functors
  //! @param __iter The iterator to transform
  //! @param __input_func The functor to apply to the iterator when reading
  //! @param __output_func The functor to apply to the iterator when writing
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_input_output_iterator(
    _Iter __current,
    _InputFn __input_func,
    _OutputFn __output_func) noexcept(::cuda::std::is_nothrow_move_constructible_v<_Iter>
                                      && ::cuda::std::is_nothrow_move_constructible_v<_InputFn>
                                      && ::cuda::std::is_nothrow_move_constructible_v<_OutputFn>)
      : __current_(::cuda::std::move(__current))
      , __input_func_(::cuda::std::in_place, ::cuda::std::move(__input_func))
      , __output_func_(::cuda::std::in_place, ::cuda::std::move(__output_func))
  {}

  //! @brief Returns a const reference to the iterator stored in this \p transform_input_output_iterator
  [[nodiscard]] _CCCL_API constexpr const _Iter& base() const& noexcept
  {
    return __current_;
  }

  //! @brief Extracts the iterator stored in this \p transform_input_output_iterator
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Iter base() && noexcept(::cuda::std::is_nothrow_move_constructible_v<_Iter>)
  {
    return ::cuda::std::move(__current_);
  }

  //! @brief Returns a proxy that transforms read values via \tparam _InputFn upon converting to \c value_type and
  //! transforms assigned values via \tparam _OutputFn before writing
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() const noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_input_output_proxy{
      __current_, const_cast<_InputFn&>(*__input_func_), const_cast<_OutputFn&>(*__output_func_)};
  }

  //! @brief Returns a proxy that transforms read values via \tparam _InputFn upon converting to \c value_type and
  //! transforms assigned values via \tparam _OutputFn before writing
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr auto operator*() noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>)
  {
    return __transform_input_output_proxy{__current_, *__input_func_, *__output_func_};
  }

  //! @brief Returns a proxy storing the current iterator advanced by \p __n that transforms read values via \tparam
  //! _InputFn upon converting to \c value_type and transforms assigned values via \tparam _OutputFn before writing
  //! @param __n The additional offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::random_access_iterator<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter2> && noexcept(__current_ + __n))
  {
    return __transform_input_output_proxy{
      __current_ + __n, const_cast<_InputFn&>(*__input_func_), const_cast<_OutputFn&>(*__output_func_)};
  }

  //! @brief Returns a proxy storing the current iterator advanced by \p __n that transforms read values via \tparam
  //! _InputFn upon converting to \c value_type and transforms assigned values via \tparam _OutputFn before writing
  //! @param __n The additional offset
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::random_access_iterator<_Iter2>)
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter2> && noexcept(__current_ + __n))
  {
    return __transform_input_output_proxy{__current_ + __n, *__input_func_, *__output_func_};
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_input_output_iterator& operator++() noexcept(noexcept(++__current_))
  {
    ++__current_;
    return *this;
  }

  //! @brief Increments the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr transform_input_output_iterator operator++(int) noexcept(
    noexcept(++__current_)
    && ::cuda::std::is_nothrow_copy_constructible_v<_Iter> && ::cuda::std::is_nothrow_copy_constructible_v<_InputFn>
    && ::cuda::std::is_nothrow_copy_constructible_v<_OutputFn>)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::bidirectional_iterator<_Iter2>)
  _CCCL_API constexpr transform_input_output_iterator& operator--() noexcept(noexcept(--__current_))
  {
    --__current_;
    return *this;
  }

  //! @brief Decrements the stored iterator
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::bidirectional_iterator<_Iter2>)
  _CCCL_API constexpr transform_input_output_iterator
  operator--(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter> && noexcept(--__current_))
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  //! @brief Advances this \c transform_input_output_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::random_access_iterator<_Iter2>)
  _CCCL_API constexpr transform_input_output_iterator&
  operator+=(difference_type __n) noexcept(noexcept(__current_ += __n))
  {
    __current_ += __n;
    return *this;
  }

  //! @brief Decrements this \c transform_input_output_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(::cuda::std::random_access_iterator<_Iter2>)
  _CCCL_API constexpr transform_input_output_iterator&
  operator-=(difference_type __n) noexcept(noexcept(__current_ -= __n))
  {
    __current_ -= __n;
    return *this;
  }

  //! @brief Compares two \c transform_input_output_iterator for equality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator==(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() == ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ == __rhs.__current_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c transform_input_output_iterator for inequality, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator!=(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() != ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::equality_comparable<_Iter2>)
  {
    return __lhs.__current_ != __rhs.__current_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Compares two \c transform_input_output_iterator for less than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ < __rhs.__current_;
  }

  //! @brief Compares two \c transform_input_output_iterator for greater than, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ > __rhs.__current_;
  }

  //! @brief Compares two \c transform_input_output_iterator for less equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ <= __rhs.__current_;
  }

  //! @brief Compares two \c transform_input_output_iterator for greater equal, directly comparing the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() < ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return __lhs.__current_ >= __rhs.__current_;
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c transform_input_output_iterator, directly three-way-comparing the stored
  //! iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() <=> ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(bool)(
      ::cuda::std::random_access_iterator<_Iter2>&& ::cuda::std::three_way_comparable<_Iter2>)
  {
    return __lhs.__current_ <=> __rhs.__current_;
  }
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  //! @brief Returns a copy of the \c transform_input_output_iterator \p __i advanced by \p __n
  //! @param __i The \c transform_input_output_iterator to advance
  //! @param __n The number of elements to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(const transform_input_output_iterator& __i,
            difference_type __n) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>
                                          && noexcept(::cuda::std::declval<const _Iter2&>() + difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_input_output_iterator)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return transform_input_output_iterator{__i.__current_ + __n, *__i.__input_func_, *__i.__output_func_};
  }

  //! @brief Returns a copy of the \c transform_input_output_iterator \p __i advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __i The \c transform_input_output_iterator to advance
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator+(difference_type __n, const transform_input_output_iterator& __i) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Iter>
    && noexcept(::cuda::std::declval<const _Iter2&>() + difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_input_output_iterator)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return transform_input_output_iterator{__i.__current_ + __n, *__i.__input_func_, *__i.__output_func_};
  }

  //! @brief Returns a copy of the the \c transform_input_output_iterator \p __i decremented by \p __n
  //! @param __i The \c transform_input_output_iterator to decrement
  //! @param __n The number of elements to decrement
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_input_output_iterator& __i,
            difference_type __n) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Iter>
                                          && noexcept(::cuda::std::declval<const _Iter2&>() - difference_type{}))
    _CCCL_TRAILING_REQUIRES(transform_input_output_iterator)(::cuda::std::random_access_iterator<_Iter2>)
  {
    return transform_input_output_iterator{__i.__current_ - __n, *__i.__input_func_, *__i.__output_func_};
  }

  //! @brief Returns the distance between \p __lhs and \p __rhs
  //! @param __lhs The left \c transform_input_output_iterator
  //! @param __rhs The right \c transform_input_output_iterator
  //! @return The distance between the stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter2 = _Iter>
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator-(const transform_input_output_iterator& __lhs, const transform_input_output_iterator& __rhs) noexcept(
    noexcept(::cuda::std::declval<const _Iter2&>() - ::cuda::std::declval<const _Iter2&>()))
    _CCCL_TRAILING_REQUIRES(difference_type)(::cuda::std::sized_sentinel_for<_Iter2, _Iter2>)
  {
    return __lhs.__current_ - __rhs.__current_;
  }
};

//! @brief make_transform_output_iterator creates a \c transform_output_iterator from an \c _Iter, an \c _InputFn, and a
//! \c _OutputFn.
//!
//! @param __iter The \c Iterator pointing to the input range of the newly created \p transform_output_iterator.
//! @param __input_fun The \c _InputFn used to transform the range pointed to by @param __iter when read
//! @param __output_fun The \c _OutputFn used to transform the range pointed to by @param __iter when written
//! @return A new \c transform_output_iterator which transforms the range at @param __iter by @param __input_fun and
//! @param __output_fun..
template <class _Iter, class _InputFn, class _OutputFn>
[[nodiscard]] _CCCL_API constexpr auto
make_transform_input_output_iterator(_Iter __iter, _InputFn __input_fun, _OutputFn __output_fun)
{
  return transform_input_output_iterator<_Iter, _InputFn, _OutputFn>{__iter, __input_fun, __output_fun};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TRANSFORM_INPUT_OUTPUT_ITERATOR_H
