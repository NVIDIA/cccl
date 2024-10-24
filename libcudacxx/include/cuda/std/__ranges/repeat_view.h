// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_REPEAT_VIEW_H
#define _LIBCUDACXX___RANGES_REPEAT_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/unreachable_sentinel.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/iota_view.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/piecewise_construct.h>
#include <cuda/std/detail/libcxx/include/tuple>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__take)
struct __fn;
_LIBCUDACXX_END_NAMESPACE_CPO

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__drop)
struct __fn;
_LIBCUDACXX_END_NAMESPACE_CPO
_LIBCUDACXX_END_NAMESPACE_VIEWS

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Tp>
_LIBCUDACXX_CONCEPT __integer_like_with_usable_difference_type =
  __signed_integer_like<_Tp> || (__integer_like<_Tp> && weakly_incrementable<_Tp>);

template <class _Tp>
using __repeat_view_iterator_difference_t = _If<__signed_integer_like<_Tp>, _Tp, _IotaDiffT<_Tp>>;

#  if _CCCL_STD_VER >= 2020
template <move_constructible _Tp, semiregular _Bound = unreachable_sentinel_t>
  requires(is_object_v<_Tp> && same_as<_Tp, remove_cv_t<_Tp>>
           && (__integer_like_with_usable_difference_type<_Bound> || same_as<_Bound, unreachable_sentinel_t>) )
#  else // ^^^ C++20 ^^^ / vvv C++20 vvv
template <
  class _Tp,
  class _Bound                                                         = unreachable_sentinel_t,
  enable_if_t<move_constructible<_Tp>, int>                            = 0,
  enable_if_t<semiregular<_Bound>, int>                                = 0,
  enable_if_t<is_object_v<_Tp> && same_as<_Tp, remove_cv_t<_Tp>>, int> = 0,
  enable_if_t<(__integer_like_with_usable_difference_type<_Bound> || same_as<_Bound, unreachable_sentinel_t>), int> = 0>
#  endif // _CCCL_STD_VER <= 2017
class repeat_view : public view_interface<repeat_view<_Tp, _Bound>>
{
  friend _CUDA_VIEWS::__take::__fn;
  friend _CUDA_VIEWS::__drop::__fn;

public:
  class __iterator
  {
    friend class repeat_view;

    using _IndexT = conditional_t<same_as<_Bound, unreachable_sentinel_t>, ptrdiff_t, _Bound>;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __iterator(const _Tp* __value, _IndexT __bound_sentinel = _IndexT())
        : __value_(__value)
        , __current_(__bound_sentinel)
    {}

  public:
    using iterator_concept  = random_access_iterator_tag;
    using iterator_category = random_access_iterator_tag;
    using value_type        = _Tp;
    using difference_type   = __repeat_view_iterator_difference_t<_IndexT>;

    _CCCL_HIDE_FROM_ABI __iterator() = default;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp& operator*() const noexcept
    {
      return *__value_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator++()
    {
      ++__current_;
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator operator++(int)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator--()
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ > 0, "The value of bound must be greater than or equal to 0");
      }
      --__current_;
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator operator--(int)
    {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n)
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ + __n >= 0, "The value of bound must be greater than or equal to 0");
      }
      __current_ += static_cast<_IndexT>(__n);
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n)
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ - __n >= 0, "The value of bound must be greater than or equal to 0");
      }
      __current_ -= static_cast<_IndexT>(__n);
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp& operator[](difference_type __n) const noexcept
    {
      return *(*this + __n);
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ == __y.__current_;
    }
#  if _CCCL_STD_VER <= 2017
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator!=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ != __y.__current_;
    }
#  endif // _CCCL_STD_VER <= 2017

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ <=> __y.__current_;
    }
#  else
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ < __y.__current_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ <= __y.__current_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ > __y.__current_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ >= __y.__current_;
    }
#  endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iterator operator+(__iterator __i, difference_type __n)
    {
      __i += __n;
      return __i;
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, __iterator __i)
    {
      __i += __n;
      return __i;
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr __iterator operator-(__iterator __i, difference_type __n)
    {
      __i -= __n;
      return __i;
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    {
      return static_cast<difference_type>(__x.__current_) - static_cast<difference_type>(__y.__current_);
    }

  private:
    const _Tp* __value_ = nullptr;
    _IndexT __current_  = _IndexT();
  };

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI repeat_view()
    requires default_initializable<_Tp>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++20 vvv
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(default_initializable<_Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr repeat_view() noexcept(is_nothrow_default_constructible_v<_Tp2>) {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES((!same_as<_Tp2, repeat_view>) _LIBCUDACXX_AND copy_constructible<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit repeat_view(const _Tp2& __value, _Bound __bound_sentinel = _Bound())
      : __value_(in_place, __value)
      , __bound_(__bound_sentinel)
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && _CCCL_TRAIT(is_signed, _Bound))
    {
      _CCCL_ASSERT(__bound_ >= 0, "The value of bound must be greater than or equal to 0");
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit repeat_view(_Tp&& __value, _Bound __bound_sentinel = _Bound())
      : __value_(in_place, _CUDA_VSTD::move(__value))
      , __bound_(__bound_sentinel)
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && _CCCL_TRAIT(is_signed, _Bound))
    {
      _CCCL_ASSERT(__bound_ >= 0, "The value of bound must be greater than or equal to 0");
    }
  }

  _LIBCUDACXX_TEMPLATE(class... _TpArgs, class... _BoundArgs)
  _LIBCUDACXX_REQUIRES(constructible_from<_Tp, _TpArgs...> _LIBCUDACXX_AND constructible_from<_Bound, _BoundArgs...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit repeat_view(
    piecewise_construct_t, tuple<_TpArgs...> __value_args, tuple<_BoundArgs...> __bound_args = tuple<>{})
      : __value_(in_place, _CUDA_VSTD::make_from_tuple<_Tp>(_CUDA_VSTD::move(__value_args)))
      , __bound_(_CUDA_VSTD::make_from_tuple<_Bound>(_CUDA_VSTD::move(__bound_args)))
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && _CCCL_TRAIT(is_signed, _Bound))
    {
      _CCCL_ASSERT(__bound_ >= 0,
                   "The behavior is undefined if Bound is not unreachable_sentinel_t and bound is negative");
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator begin() const
  {
    return __iterator(_CUDA_VSTD::addressof(*__value_));
  }

  _LIBCUDACXX_TEMPLATE(class _Bound2 = _Bound)
  _LIBCUDACXX_REQUIRES((!same_as<_Bound2, unreachable_sentinel_t>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator end() const
  {
    return __iterator(_CUDA_VSTD::addressof(*__value_), __bound_);
  }

  _LIBCUDACXX_TEMPLATE(class _Bound2 = _Bound)
  _LIBCUDACXX_REQUIRES(same_as<_Bound2, unreachable_sentinel_t>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr unreachable_sentinel_t end() const noexcept
  {
    return unreachable_sentinel;
  }

  _LIBCUDACXX_TEMPLATE(class _Bound2 = _Bound)
  _LIBCUDACXX_REQUIRES((!same_as<_Bound2, unreachable_sentinel_t>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    return _CUDA_VSTD::__to_unsigned_like(__bound_);
  }

private:
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Tp> __value_;
  _CCCL_NO_UNIQUE_ADDRESS _Bound __bound_ = _Bound();
};

template <class _Tp, class _Bound>
_CCCL_HOST_DEVICE repeat_view(_Tp, _Bound) -> repeat_view<_Tp, _Bound>;

_LIBCUDACXX_END_NAMESPACE_RANGES

// clang-format off
_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__repeat)
struct __fn {
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __value) const
    noexcept(noexcept(ranges::repeat_view(_CUDA_VSTD::forward<_Tp>(__value))))
    -> repeat_view<remove_cvref_t<_Tp>>
    { return          ranges::repeat_view(_CUDA_VSTD::forward<_Tp>(__value)); }


  template <class _Tp, class _Bound>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __value, _Bound&& __bound_sentinel) const
    noexcept(noexcept(ranges::repeat_view(_CUDA_VSTD::forward<_Tp>(__value), _CUDA_VSTD::forward<_Bound>(__bound_sentinel))))
    -> repeat_view<remove_cvref_t<_Tp>, remove_cvref_t<_Bound>>
    { return          ranges::repeat_view(_CUDA_VSTD::forward<_Tp>(__value), _CUDA_VSTD::forward<_Bound>(__bound_sentinel)); }
};
_LIBCUDACXX_END_NAMESPACE_CPO
// clang-format on

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto repeat = __repeat::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_repeat_specialization = false;

template <class _Tp, class _Bound>
_CCCL_INLINE_VAR constexpr bool __is_repeat_specialization<repeat_view<_Tp, _Bound>> = true;

_LIBCUDACXX_END_NAMESPACE_RANGES

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___RANGES_REPEAT_VIEW_H
