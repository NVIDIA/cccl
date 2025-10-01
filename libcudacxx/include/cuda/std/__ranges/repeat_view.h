// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_REPEAT_VIEW_H
#define _CUDA_STD___RANGES_REPEAT_VIEW_H

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

#include <cuda/std/__cccl/prologue.h>

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_CCCL_BEGIN_NAMESPACE_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__take)
struct __fn;
_CCCL_END_NAMESPACE_CPO
_CCCL_END_NAMESPACE_VIEWS

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _Tp>
_CCCL_CONCEPT __integer_like_with_usable_difference_type =
  __signed_integer_like<_Tp> || (__integer_like<_Tp> && weakly_incrementable<_Tp>);

template <class _Tp>
using __repeat_view_iterator_difference_t _CCCL_NODEBUG_ALIAS = _If<__signed_integer_like<_Tp>, _Tp, _IotaDiffT<_Tp>>;

#if _CCCL_HAS_CONCEPTS()
template <move_constructible _Tp, semiregular _Bound = unreachable_sentinel_t>
  requires(is_object_v<_Tp> && same_as<_Tp, remove_cv_t<_Tp>>
           && (__integer_like_with_usable_difference_type<_Bound> || same_as<_Bound, unreachable_sentinel_t>) )
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <
  class _Tp,
  class _Bound                                                         = unreachable_sentinel_t,
  enable_if_t<move_constructible<_Tp>, int>                            = 0,
  enable_if_t<semiregular<_Bound>, int>                                = 0,
  enable_if_t<is_object_v<_Tp> && same_as<_Tp, remove_cv_t<_Tp>>, int> = 0,
  enable_if_t<(__integer_like_with_usable_difference_type<_Bound> || same_as<_Bound, unreachable_sentinel_t>), int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class repeat_view : public view_interface<repeat_view<_Tp, _Bound>>
{
  friend ::cuda::std::ranges::views::__take::__fn;

public:
  class __iterator
  {
    friend class repeat_view;

    using _IndexT _CCCL_NODEBUG_ALIAS = conditional_t<same_as<_Bound, unreachable_sentinel_t>, ptrdiff_t, _Bound>;

    _CCCL_API constexpr explicit __iterator(const _Tp* __value, _IndexT __bound_sentinel = _IndexT()) noexcept(
      is_nothrow_copy_constructible_v<_IndexT>)
        : __value_(__value)
        , __current_(__bound_sentinel)
    {}

  public:
    using iterator_concept  = random_access_iterator_tag;
    using iterator_category = random_access_iterator_tag;
    using value_type        = _Tp;
    using difference_type   = __repeat_view_iterator_difference_t<_IndexT>;

    _CCCL_HIDE_FROM_ABI __iterator() = default;

    [[nodiscard]] _CCCL_API constexpr const _Tp& operator*() const noexcept
    {
      return *__value_;
    }

    _CCCL_API constexpr __iterator& operator++()
    {
      ++__current_;
      return *this;
    }

    _CCCL_API constexpr __iterator operator++(int)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _CCCL_API constexpr __iterator& operator--()
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ > 0, "The value of bound must be greater than or equal to 0");
      }
      --__current_;
      return *this;
    }

    _CCCL_API constexpr __iterator operator--(int)
    {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    _CCCL_API constexpr __iterator& operator+=(difference_type __n)
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ + __n >= 0, "The value of bound must be greater than or equal to 0");
      }
      __current_ += static_cast<_IndexT>(__n);
      return *this;
    }

    _CCCL_API constexpr __iterator& operator-=(difference_type __n)
    {
      if constexpr (!same_as<_Bound, unreachable_sentinel_t>)
      {
        _CCCL_ASSERT(__current_ - __n >= 0, "The value of bound must be greater than or equal to 0");
      }
      __current_ -= static_cast<_IndexT>(__n);
      return *this;
    }

    [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](difference_type __n) const noexcept
    {
      return *(*this + __n);
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ == __y.__current_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ != __y.__current_;
    }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    [[nodiscard]] _CCCL_API friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ <=> __y.__current_;
    }
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR()

    [[nodiscard]] _CCCL_API friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ < __y.__current_;
    }
    [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ <= __y.__current_;
    }
    [[nodiscard]] _CCCL_API friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ > __y.__current_;
    }
    [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ >= __y.__current_;
    }

    [[nodiscard]] _CCCL_API friend constexpr __iterator operator+(__iterator __i, difference_type __n)
    {
      __i += __n;
      return __i;
    }

    [[nodiscard]] _CCCL_API friend constexpr __iterator operator+(difference_type __n, __iterator __i)
    {
      __i += __n;
      return __i;
    }

    [[nodiscard]] _CCCL_API friend constexpr __iterator operator-(__iterator __i, difference_type __n)
    {
      __i -= __n;
      return __i;
    }

    [[nodiscard]] _CCCL_API friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    {
      return static_cast<difference_type>(__x.__current_) - static_cast<difference_type>(__y.__current_);
    }

  private:
    const _Tp* __value_ = nullptr;
    _IndexT __current_  = _IndexT();
  };

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI repeat_view()
    requires default_initializable<_Tp>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(default_initializable<_Tp2>)
  _CCCL_API constexpr repeat_view() noexcept(is_nothrow_default_constructible_v<_Tp2>) {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES((!same_as<_Tp2, repeat_view>) _CCCL_AND copy_constructible<_Tp2>)
  _CCCL_API constexpr explicit repeat_view(const _Tp2& __value, _Bound __bound_sentinel = _Bound())
      : __value_(in_place, __value)
      , __bound_(__bound_sentinel)
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && is_signed_v<_Bound>)
    {
      _CCCL_ASSERT(__bound_ >= 0, "The value of bound must be greater than or equal to 0");
    }
  }

  _CCCL_API constexpr explicit repeat_view(_Tp&& __value, _Bound __bound_sentinel = _Bound())
      : __value_(in_place, ::cuda::std::move(__value))
      , __bound_(__bound_sentinel)
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && is_signed_v<_Bound>)
    {
      _CCCL_ASSERT(__bound_ >= 0, "The value of bound must be greater than or equal to 0");
    }
  }

  _CCCL_TEMPLATE(class... _TpArgs, class... _BoundArgs)
  _CCCL_REQUIRES(constructible_from<_Tp, _TpArgs...> _CCCL_AND constructible_from<_Bound, _BoundArgs...>)
  _CCCL_API constexpr explicit repeat_view(
    piecewise_construct_t, tuple<_TpArgs...> __value_args, tuple<_BoundArgs...> __bound_args = tuple<>{})
      : __value_(in_place, ::cuda::std::make_from_tuple<_Tp>(::cuda::std::move(__value_args)))
      , __bound_(::cuda::std::make_from_tuple<_Bound>(::cuda::std::move(__bound_args)))
  {
    if constexpr (!same_as<_Bound, unreachable_sentinel_t> && is_signed_v<_Bound>)
    {
      _CCCL_ASSERT(__bound_ >= 0,
                   "The behavior is undefined if Bound is not unreachable_sentinel_t and bound is negative");
    }
  }

  [[nodiscard]] _CCCL_API constexpr __iterator begin() const
  {
    return __iterator(::cuda::std::addressof(*__value_));
  }

  _CCCL_API constexpr auto end() const noexcept(is_nothrow_copy_constructible_v<_Bound>)
  {
    if constexpr (same_as<_Bound, unreachable_sentinel_t>)
    {
      return unreachable_sentinel;
    }
    else
    {
      return __iterator(::cuda::std::addressof(*__value_), __bound_);
    }
  }

  _CCCL_TEMPLATE(class _Bound2 = _Bound)
  _CCCL_REQUIRES((!same_as<_Bound2, unreachable_sentinel_t>) )
  _CCCL_API constexpr auto size() const
  {
    return ::cuda::std::__to_unsigned_like(__bound_);
  }

private:
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Tp> __value_;
  _CCCL_NO_UNIQUE_ADDRESS _Bound __bound_ = _Bound();
};

template <class _Tp, class _Bound>
_CCCL_HOST_DEVICE repeat_view(_Tp, _Bound) -> repeat_view<_Tp, _Bound>;

_CCCL_END_NAMESPACE_RANGES

// clang-format off
_CCCL_BEGIN_NAMESPACE_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__repeat)
struct __fn {
  template <class _Tp>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __value) const
    noexcept(noexcept(ranges::repeat_view(::cuda::std::forward<_Tp>(__value))))
    -> repeat_view<remove_cvref_t<_Tp>>
    { return          ranges::repeat_view(::cuda::std::forward<_Tp>(__value)); }


  template <class _Tp, class _Bound>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __value, _Bound&& __bound_sentinel) const
    noexcept(noexcept(ranges::repeat_view(::cuda::std::forward<_Tp>(__value), ::cuda::std::forward<_Bound>(__bound_sentinel))))
    -> repeat_view<remove_cvref_t<_Tp>, remove_cvref_t<_Bound>>
    { return          ranges::repeat_view(::cuda::std::forward<_Tp>(__value), ::cuda::std::forward<_Bound>(__bound_sentinel)); }
};
_CCCL_END_NAMESPACE_CPO
// clang-format on

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto repeat = __repeat::__fn{};
} // namespace __cpo
_CCCL_END_NAMESPACE_VIEWS

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _Tp>
inline constexpr bool __is_repeat_specialization = false;

template <class _Tp, class _Bound>
inline constexpr bool __is_repeat_specialization<repeat_view<_Tp, _Bound>> = true;

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_REPEAT_VIEW_H
