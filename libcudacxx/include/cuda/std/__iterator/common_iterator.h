// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_COMMON_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_COMMON_ITERATOR_H

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
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
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
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/cstdlib>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#  if _CCCL_STD_VER >= 2020
template <class _Iter>
concept __use_postfix_proxy =
  !requires(_Iter& __it) {
    { *__it++ } -> __can_reference;
  } && indirectly_readable<_Iter> && constructible_from<iter_value_t<_Iter>, iter_reference_t<_Iter>>
  && move_constructible<iter_value_t<_Iter>>;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __postfix_can_reference_,
  requires(_Iter& __it)(requires(__dereferenceable<decltype(__it++)>), requires(__can_reference<decltype(*__it++)>)));

template <class _Iter>
_LIBCUDACXX_CONCEPT __postfix_can_reference = _LIBCUDACXX_FRAGMENT(__postfix_can_reference_, _Iter);

template <class _Iter>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __use_postfix_proxy_,
  requires()(requires(!__postfix_can_reference<_Iter>),
             requires(indirectly_readable<_Iter>),
             requires(constructible_from<iter_value_t<_Iter>, iter_reference_t<_Iter>>),
             requires(move_constructible<iter_value_t<_Iter>>)));

template <class _Iter>
_LIBCUDACXX_CONCEPT __use_postfix_proxy = _LIBCUDACXX_FRAGMENT(__use_postfix_proxy_, _Iter);
#  endif // _CCCL_STD_VER <= 2017

#  if _CCCL_STD_VER >= 2020
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent>
  requires(!same_as<_Iter, _Sent> && copyable<_Iter>)
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter,
          class _Sent,
          enable_if_t<input_or_output_iterator<_Iter>, int>             = 0,
          enable_if_t<sentinel_for<_Sent, _Iter>, int>                  = 0,
          enable_if_t<(!same_as<_Iter, _Sent> && copyable<_Iter>), int> = 0>
#  endif // _CCCL_STD_VER <= 2017
class common_iterator
{
  struct __proxy
  {
    _LIBCUDACXX_HIDE_FROM_ABI constexpr const iter_value_t<_Iter>* operator->() const noexcept
    {
      return _CUDA_VSTD::addressof(__value_);
    }
    iter_value_t<_Iter> __value_;
  };

  struct __postfix_proxy
  {
    _LIBCUDACXX_HIDE_FROM_ABI constexpr const iter_value_t<_Iter>& operator*() const noexcept
    {
      return __value_;
    }
    iter_value_t<_Iter> __value_;
  };

public:
  __variant_like<_Iter, _Sent> __hold_;

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI constexpr common_iterator()
    requires default_initializable<_Iter>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
  _LIBCUDACXX_REQUIRES(default_initializable<_I2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator() noexcept(is_nothrow_default_constructible_v<_I2>) {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator(_Iter __i) noexcept(is_nothrow_move_constructible_v<_Iter>)
      : __hold_(__construct_first{}, _CUDA_VSTD::move(__i))
  {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator(_Sent __s) noexcept(is_nothrow_move_constructible_v<_Sent>)
      : __hold_(__construct_second{}, _CUDA_VSTD::move(__s))
  {}

  _LIBCUDACXX_TEMPLATE(class _I2, class _S2)
  _LIBCUDACXX_REQUIRES(convertible_to<const _I2&, _Iter>&& convertible_to<const _S2&, _Sent>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator(const common_iterator<_I2, _S2>& __other) noexcept(
    is_nothrow_constructible_v<_Iter, const _I2&> && is_nothrow_constructible_v<_Sent, const _S2&>)
      : __hold_(__other.__get_hold())
  {}

  _LIBCUDACXX_TEMPLATE(class _I2, class _S2)
  _LIBCUDACXX_REQUIRES(
    convertible_to<const _I2&, _Iter> _LIBCUDACXX_AND convertible_to<const _S2&, _Sent> _LIBCUDACXX_AND
      assignable_from<_Iter&, const _I2&> _LIBCUDACXX_AND assignable_from<_Sent&, const _S2&>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator& operator=(const common_iterator<_I2, _S2>& __other)
  {
    __hold_ = __other.__get_hold();
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*()
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    return *__hold_.__first_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
  _LIBCUDACXX_REQUIRES(__dereferenceable<const _I2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    return *__hold_.__first_;
  }

  _LIBCUDACXX_TEMPLATE(class _I2 = _Iter)
  _LIBCUDACXX_REQUIRES(indirectly_readable<const _I2> _LIBCUDACXX_AND(
    __has_const_arrow<_I2> || is_reference_v<iter_reference_t<_I2>>
    || constructible_from<iter_value_t<_I2>, iter_reference_t<_I2>>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator->() const
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to dereference a non-dereferenceable common_iterator");
    if constexpr (__has_const_arrow<_Iter>)
    {
      return __hold_.__first_;
    }
    else if constexpr (is_reference_v<iter_reference_t<_Iter>>)
    {
      auto&& __tmp = *__hold_.__first_;
      return _CUDA_VSTD::addressof(__tmp);
    }
    else
    {
      return __proxy{*__hold_.__first_};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr common_iterator& operator++()
  {
    _CCCL_ASSERT(__hold_.__holds_first(), "Attempted to increment a non-dereferenceable common_iterator");
    ++__hold_.__first_;
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator++(int)
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

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator==(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    sentinel_for<_S2, _Iter>&& sentinel_for<_Sent, _I2> && (!equality_comparable_with<_Iter, _I2>) )
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(!__x.__hold_.valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    auto __x_contains = __x.__hold_.__contains_;
    auto __y_contains = __y_hold.__contains_;

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
#  if _CCCL_STD_VER < 2020
  template <class _I2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const common_iterator& __x, const common_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)((!equality_comparable<_I2>) )
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(!__x.__hold_.valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    auto __x_contains = __x.__hold_.__contains_;
    auto __y_contains = __y_hold.__contains_;

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

  template <class _I2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const common_iterator& __x, const common_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)((!equality_comparable<_I2>) )
  {
    return !(__x == __y);
  }

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator!=(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    sentinel_for<_S2, _Iter>&& sentinel_for<_Sent, _I2> && (!equality_comparable_with<_Iter, _I2>) )
  {
    return !(__x == __y);
  }
#  endif

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator==(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    sentinel_for<_S2, _Iter>&& sentinel_for<_Sent, _I2>&& equality_comparable_with<_Iter, _I2>)
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(!__x.__hold_.valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return true;
    }
    else if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__first_;
    }
    else if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }
    else
    {
      return __x.__hold_.__second_ == __y_hold.__first_;
    }
  }
#  if _CCCL_STD_VER < 2020
  template <class _I2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const common_iterator& __x, const common_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<_I2>)
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(!__x.__hold_.valueless_by_exception(), "Attempted to compare a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.valueless_by_exception(), "Attempted to compare a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return true;
    }
    else if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__first_;
    }
    else if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ == __y_hold.__second_;
    }
    else
    {
      return __x.__hold_.__second_ == __y_hold.__first_;
    }
  }

  template <class _I2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const common_iterator& __x, const common_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<_I2>)
  {
    return !(__x == __y);
  }

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator!=(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) _LIBCUDACXX_TRAILING_REQUIRES(bool)(
    sentinel_for<_S2, _Iter>&& sentinel_for<_Sent, _I2>&& equality_comparable_with<_Iter, _I2>)
  {
    return !(__x == __y);
  }
#  endif

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator-(const common_iterator& __x, const common_iterator<_I2, _S2>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(iter_difference_t<_I2>)(
      sized_sentinel_for<_I2, _Iter>&& sized_sentinel_for<_S2, _Iter>&& sized_sentinel_for<_Sent, _I2>)
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(!__x.__hold_.valueless_by_exception(), "Attempted to subtract from a valueless common_iterator");
    _CCCL_ASSERT(!__y_hold.valueless_by_exception(), "Attempted to subtract a valueless common_iterator");

    if (__x.__hold_.__holds_second() && __y_hold.__holds_second())
    {
      return 0;
    }
    else if (__x.__hold_.__holds_first() && __y_hold.__holds_first())
    {
      return __x.__hold_.__first_ - __y_hold.__first_;
    }
    else if (__x.__hold_.__holds_first())
    {
      return __x.__hold_.__first_ - __y_hold.__second_;
    }
    else
    {
      return __x.__hold_.__second_ - __y_hold.__first_;
    }
  }

  template <class _Iter2 = _Iter>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  iter_move(const common_iterator& __i) noexcept(noexcept(_CUDA_VRANGES::iter_move(declval<const _Iter&>())))
    _LIBCUDACXX_TRAILING_REQUIRES(iter_rvalue_reference_t<_Iter2>)(input_iterator<_Iter2>)
  {
    _CCCL_ASSERT(__i.__hold_.__holds_first(), "Attempted to iter_move a non-dereferenceable common_iterator");
    return _CUDA_VRANGES::iter_move(__i.__hold_.__first_);
  }

  template <class _I2, class _S2>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  iter_swap(const common_iterator& __x, const common_iterator<_I2, _S2>& __y) noexcept(
    noexcept(_CUDA_VRANGES::iter_swap(declval<const _Iter&>(), declval<const _I2&>())))
    _LIBCUDACXX_TRAILING_REQUIRES(void)(indirectly_swappable<_I2, _Iter>)
  {
    auto& __y_hold = __y.__get_hold();
    _CCCL_ASSERT(__x.__hold_.__holds_first(), "Attempted to iter_swap a non-dereferenceable common_iterator");
    _CCCL_ASSERT(__y_hold.__holds_first(), "Attempted to iter_swap a non-dereferenceable common_iterator");
    return _CUDA_VRANGES::iter_swap(__x.__hold_.__first_, __y_hold.__first_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __variant_like<_Iter, _Sent>& __get_hold() noexcept
  {
    return __hold_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr const __variant_like<_Iter, _Sent>& __get_hold() const noexcept
  {
    return __hold_;
  }
};

template <class _Iter, class _Sent>
struct incrementable_traits<common_iterator<_Iter, _Sent>>
{
  using difference_type = iter_difference_t<_Iter>;
};

#  if _CCCL_STD_VER >= 2020
template <class _Iter>
concept __denotes_forward_iter = requires {
  typename iterator_traits<_Iter>::iterator_category;
} && derived_from<typename iterator_traits<_Iter>::iterator_category, forward_iterator_tag>;

template <class _Iter, class _Sent>
concept __common_iter_has_ptr_op = requires(const common_iterator<_Iter, _Sent>& __a) { __a.operator->(); };
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __denotes_forward_iter_,
  requires()(typename(typename iterator_traits<_Iter>::iterator_category),
             requires(derived_from<typename iterator_traits<_Iter>::iterator_category, forward_iterator_tag>)));

template <class _Iter>
_LIBCUDACXX_CONCEPT __denotes_forward_iter = _LIBCUDACXX_FRAGMENT(__denotes_forward_iter_, _Iter);

template <class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT_FRAGMENT(__common_iter_has_ptr_op_,
                             requires(const common_iterator<_Iter, _Sent>& __i)((__i.operator->())));

template <class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT __common_iter_has_ptr_op = _LIBCUDACXX_FRAGMENT(__common_iter_has_ptr_op_, _Iter, _Sent);
#  endif // _CCCL_STD_VER <= 2017

template <class, class, class = void>
struct __arrow_type_or_void
{
  using type = void;
};

template <class _Iter, class _Sent>
struct __arrow_type_or_void<_Iter, _Sent, enable_if_t<__common_iter_has_ptr_op<_Iter, _Sent>>>
{
  using type = decltype(declval<const common_iterator<_Iter, _Sent>&>().operator->());
};

#  if _CCCL_STD_VER >= 2020
template <input_iterator _Iter, class _Sent>
struct iterator_traits<common_iterator<_Iter, _Sent>>
{
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter, class _Sent>
struct iterator_traits<common_iterator<_Iter, _Sent>, enable_if_t<input_iterator<_Iter>>>
{
#  endif // _CCCL_STD_VER <= 2017
  using iterator_concept  = _If<forward_iterator<_Iter>, forward_iterator_tag, input_iterator_tag>;
  using iterator_category = _If<__denotes_forward_iter<_Iter>, forward_iterator_tag, input_iterator_tag>;
  using pointer           = typename __arrow_type_or_void<_Iter, _Sent>::type;
  using value_type        = iter_value_t<_Iter>;
  using difference_type   = iter_difference_t<_Iter>;
  using reference         = iter_reference_t<_Iter>;
};

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_COMMON_ITERATOR_H
