// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_COMMON_ITERATOR_H
#define _CUDA_STD___ITERATOR_COMMON_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/common_iterator.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__iterator/variant_like.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

template <class _Iter>
_CCCL_CONCEPT __postfix_can_reference = _CCCL_REQUIRES_EXPR((_Iter), _Iter& __it)(_Satisfies(__can_reference) * __it++);

template <class _Iter>
_CCCL_CONCEPT __use_postfix_proxy = _CCCL_REQUIRES_EXPR((_Iter), )(
  requires(!__postfix_can_reference<_Iter>),
  requires(indirectly_readable<_Iter>),
  requires(constructible_from<iter_value_t<_Iter>, iter_reference_t<_Iter>>),
  requires(move_constructible<iter_value_t<_Iter>>));

#if _CCCL_HAS_CONCEPTS()
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent>
  requires(!same_as<_Iter, _Sent> && copyable<_Iter>)
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Iter,
          class _Sent,
          enable_if_t<input_or_output_iterator<_Iter>, int>,
          enable_if_t<sentinel_for<_Sent, _Iter>, int>,
          enable_if_t<(!same_as<_Iter, _Sent> && copyable<_Iter>), int>>
#endif // !_CCCL_HAS_CONCEPTS()
class _CCCL_TYPE_VISIBILITY_DEFAULT common_iterator
{
public:
  // These should be private but older compilers (gcc-7 and clang-14) complain about implicitly
  // naming private types in decltype or auto expressions.
  struct __proxy
  {
    iter_value_t<_Iter> __value_;

    [[nodiscard]] _CCCL_API constexpr const iter_value_t<_Iter>* operator->() const noexcept
    {
      return ::cuda::std::addressof(__value_);
    }
  };

  struct __postfix_proxy
  {
    iter_value_t<_Iter> __value_;

    [[nodiscard]] _CCCL_API constexpr const iter_value_t<_Iter>& operator*() const noexcept
    {
      return __value_;
    }
  };

  __variant_like<_Iter, _Sent> __hold_;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI constexpr common_iterator() noexcept(is_nothrow_default_constructible_v<_Iter>)
    requires default_initializable<_Iter>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(default_initializable<_I2>)
  _CCCL_API constexpr common_iterator() noexcept(is_nothrow_default_constructible_v<_I2>) {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_HIDE_FROM_ABI constexpr common_iterator(const common_iterator&)            = default;
  _CCCL_HIDE_FROM_ABI constexpr common_iterator& operator=(const common_iterator&) = default;
  _CCCL_HIDE_FROM_ABI constexpr common_iterator(common_iterator&&)                 = default;
  _CCCL_HIDE_FROM_ABI constexpr common_iterator& operator=(common_iterator&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr common_iterator(_Iter __i) noexcept(is_nothrow_move_constructible_v<_Iter>)
      : __hold_{__construct_first{}, ::cuda::std::move(__i)}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr common_iterator(_Sent __s) noexcept(is_nothrow_move_constructible_v<_Sent>)
      : __hold_{__construct_second{}, ::cuda::std::move(__s)}
  {}

  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(convertible_to<const _I2&, _Iter> _CCCL_AND convertible_to<const _S2&, _Sent>)
  _CCCL_API constexpr common_iterator(const common_iterator<_I2, _S2>& __other) noexcept(
    is_nothrow_constructible_v<_Iter, const _I2&> && is_nothrow_constructible_v<_Sent, const _S2&>)
      : __hold_{__other.__get_hold()}
  {}

  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(convertible_to<const _I2&, _Iter> _CCCL_AND convertible_to<const _S2&, _Sent> _CCCL_AND
                   assignable_from<_Iter&, const _I2&> _CCCL_AND assignable_from<_Sent&, const _S2&>)
  _CCCL_API constexpr common_iterator& operator=(const common_iterator<_I2, _S2>& __other)
  {
    __hold_ = __other.__get_hold();
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*()
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    return *__hold_.__first_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(__dereferenceable<const _I2>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    return *__hold_.__first_;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(indirectly_readable<const _I2> _CCCL_AND(
    __has_const_arrow<_I2> || is_reference_v<iter_reference_t<_I2>>
    || constructible_from<iter_value_t<_I2>, iter_reference_t<_I2>>))
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator->() const
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    if constexpr (__has_const_arrow<_Iter>)
    {
      return __hold_.__first_;
    }
    else if constexpr (is_reference_v<iter_reference_t<_Iter>>)
    {
      auto&& __tmp = *__hold_.__first_;

      return ::cuda::std::addressof(__tmp);
    }
    else
    {
      return __proxy{*__hold_.__first_};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr common_iterator& operator++()
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to increment a non-dereferenceable common_iterator");
    ++__hold_.__first_;
    return *this;
  }

  _CCCL_API constexpr decltype(auto) operator++(int)
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to increment a non-dereferenceable common_iterator");
    if constexpr (forward_iterator<_Iter>)
    {
      auto __tmp = *this;

      ++*this;
      return __tmp;
    }
    else if constexpr (__use_postfix_proxy<_Iter>)
    {
      auto __p = __postfix_proxy{**this};

      ++*this;
      return __p;
    }
    else
    {
      return __hold_.__first_++;
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(
    sentinel_for<_S2, _Iter> _CCCL_AND sentinel_for<_Sent, _I2> _CCCL_AND(!equality_comparable_with<_Iter, _I2>))
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(!__x.__hold_.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    const auto __x_contains = __x.__hold_.__contains_;
    const auto __y_contains = __y_hold.__contains_;

    if (__x_contains == __y_contains)
    {
      return true;
    }

    if (__x_contains == __variant_like_state::__holds_first)
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }

    return __x.__hold_.__second_ == __y_hold.__first_;
  }

#if _CCCL_STD_VER <= 2017
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES((!equality_comparable<_I2>) )
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const common_iterator& __x, const common_iterator& __y)
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(!__x.__hold_.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    const auto __x_contains = __x.__hold_.__contains_;
    const auto __y_contains = __y_hold.__contains_;

    if (__x_contains == __y_contains)
    {
      return true;
    }

    if (__x_contains == __variant_like_state::__holds_first)
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }

    return __x.__hold_.__second_ == __y_hold.__first_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES((!equality_comparable<_I2>) )
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const common_iterator& __x, const common_iterator& __y)
  {
    return !(__x == __y);
  }

  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(
    sentinel_for<_S2, _Iter> _CCCL_AND sentinel_for<_Sent, _I2> _CCCL_AND(!equality_comparable_with<_Iter, _I2>))
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(
    sentinel_for<_S2, _Iter> _CCCL_AND sentinel_for<_Sent, _I2> _CCCL_AND equality_comparable_with<_Iter, _I2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(!__x.__hold_.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return true;
    }

    if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__first_;
    }

    if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }

    return __x.__hold_.__second_ == __y_hold.__first_;
  }
#if _CCCL_STD_VER <= 2017
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(equality_comparable<_I2>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const common_iterator& __x, const common_iterator& __y)
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(!__x.__hold_.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.__valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return true;
    }

    if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__first_;
    }

    if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }

    return __x.__hold_.__second_ == __y_hold.__first_;
  }

  _CCCL_TEMPLATE(class _I2 = _Iter)
  _CCCL_REQUIRES(equality_comparable<_I2>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const common_iterator& __x, const common_iterator& __y)
  {
    return !(__x == __y);
  }

  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(
    sentinel_for<_S2, _Iter> _CCCL_AND sentinel_for<_Sent, _I2> _CCCL_AND equality_comparable_with<_Iter, _I2>)
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
  {
    return !(__x == __y);
  }
#endif // _CCCL_STD_VER <= 2017

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(
    sized_sentinel_for<_I2, _Iter> _CCCL_AND sized_sentinel_for<_S2, _Iter> _CCCL_AND sized_sentinel_for<_Sent, _I2>)
  [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_I2>
  operator-(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(!__x.__hold_.__valueless_by_exception(), "Attempted to subtract from a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.__valueless_by_exception(), "Attempted to subtract a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return 0;
    }

    if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ - __y_hold.__first_;
    }

    if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ - __y_hold.__second_;
    }

    return __x.__hold_.__second_ - __y_hold.__first_;
  }

  _CCCL_TEMPLATE(class _Iter2 = _Iter)
  _CCCL_REQUIRES(input_iterator<_Iter2>)
  _CCCL_API friend constexpr iter_rvalue_reference_t<_Iter2>
  iter_move(const common_iterator& __i) noexcept(noexcept(::cuda::std::ranges::iter_move(declval<const _Iter&>())))
  {
    _CCCL_ASSERT(__i.__hold_.__holds_first(), "Attempted to iter_move a non-dereferenceable common_iterator");
    return ::cuda::std::ranges::iter_move(__i.__hold_.__first_);
  }

  _CCCL_TEMPLATE(class _I2, class _S2)
  _CCCL_REQUIRES(indirectly_swappable<_I2, _Iter>)
  _CCCL_API friend constexpr void iter_swap(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) noexcept(
    noexcept(::cuda::std::ranges::iter_swap(::cuda::std::declval<const _Iter&>(), ::cuda::std::declval<const _I2&>())))
  {
    const auto& __y_hold = __y.__get_hold();

    _CCCL_ASSERT(__x.__hold_.__holds_first(), "Attempted to iter_swap a non-dereferenceable common_iterator");
    _CCCL_ASSERT(__y_hold.__holds_first(), "Attempted to iter_swap a non-dereferenceable common_iterator");
    return ::cuda::std::ranges::iter_swap(__x.__hold_.__first_, __y_hold.__first_);
  }

  [[nodiscard]] _CCCL_API constexpr __variant_like<_Iter, _Sent>& __get_hold() noexcept
  {
    return __hold_;
  }
  [[nodiscard]] _CCCL_API constexpr const __variant_like<_Iter, _Sent>& __get_hold() const noexcept
  {
    return __hold_;
  }
};

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(common_iterator)

template <class _Iter, class _Sent>
struct incrementable_traits<common_iterator<_Iter, _Sent>>
{
  using difference_type = iter_difference_t<_Iter>;
};

#if _CCCL_COMPILER(GCC) // GCC breaks with a circular definition here
template <class _Iter, class _Sent>
struct __is_primary_std_template<common_iterator<_Iter, _Sent>> : true_type
{};
#endif // _CCCL_COMPILER(GCC)

template <class _Iter, class _Sent>
struct iterator_traits<common_iterator<_Iter, _Sent>, enable_if_t<input_iterator<_Iter>>>
{
  using iterator_concept = conditional_t<forward_iterator<_Iter>, forward_iterator_tag, input_iterator_tag>;
  using iterator_category =
    conditional_t<__has_iterator_category_convertible_to<_Iter, forward_iterator_tag>,
                  forward_iterator_tag,
                  input_iterator_tag>;
  using pointer         = __iterator_traits_member_pointer_or_arrow_or_void<common_iterator<_Iter, _Sent>>;
  using value_type      = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;
  using reference       = iter_reference_t<_Iter>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_COMMON_ITERATOR_H
