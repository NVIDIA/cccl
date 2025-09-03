// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_IOTA_VIEW_H
#define _CUDA_STD___RANGES_IOTA_VIEW_H

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
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/unreachable_sentinel.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <weakly_incrementable _Start, semiregular _BoundSentinel = unreachable_sentinel_t>
  requires __weakly_equality_comparable_with<_Start, _BoundSentinel> && copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          class _BoundSentinel                                                        = unreachable_sentinel_t,
          enable_if_t<weakly_incrementable<_Start>, int>                              = 0,
          enable_if_t<semiregular<_BoundSentinel>, int>                               = 0,
          enable_if_t<__weakly_equality_comparable_with<_Start, _BoundSentinel>, int> = 0,
          enable_if_t<copyable<_Start>, int>                                          = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class iota_view : public view_interface<iota_view<_Start, _BoundSentinel>>
{
public:
  using __iterator = ::cuda::counting_iterator<_Start>;

  struct __sentinel
  {
    friend class iota_view;

  private:
    _BoundSentinel __bound_sentinel_ = _BoundSentinel();

  public:
    _CCCL_HIDE_FROM_ABI __sentinel() = default;
    _CCCL_API constexpr explicit __sentinel(_BoundSentinel __bound_sentinel)
        : __bound_sentinel_(::cuda::std::move(__bound_sentinel))
    {}

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __iterator& __x, const __sentinel& __y)
    {
      return __x.__value_ == __y.__bound_sentinel_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __sentinel& __x, const __iterator& __y)
    {
      return __x.__bound_sentinel_ == __y.__value_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator& __x, const __sentinel& __y)
    {
      return __x.__value_ != __y.__bound_sentinel_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __sentinel& __x, const __iterator& __y)
    {
      return __x.__bound_sentinel_ != __y.__value_;
    }
#endif // _CCCL_STD_VER <= 2017

    _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
    _CCCL_REQUIRES(sized_sentinel_for<_BoundSentinel2, _Start>)
    [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Start>
    operator-(const __iterator& __x, const __sentinel& __y)
    {
      return __x.__value_ - __y.__bound_sentinel_;
    }

    _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
    _CCCL_REQUIRES(sized_sentinel_for<_BoundSentinel2, _Start>)
    [[nodiscard]] _CCCL_API friend constexpr iter_difference_t<_Start>
    operator-(const __sentinel& __x, const __iterator& __y)
    {
      return -(__y - __x);
    }
  };

private:
  _Start __value_                  = _Start();
  _BoundSentinel __bound_sentinel_ = _BoundSentinel();

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI iota_view()
    requires default_initializable<_Start>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Start2 = _Start)
  _CCCL_REQUIRES(default_initializable<_Start2>)
  _CCCL_API constexpr iota_view() noexcept(is_nothrow_default_constructible_v<_Start2>)
      : view_interface<iota_view<_Start, _BoundSentinel>>()
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_API constexpr explicit iota_view(_Start __value) noexcept(is_nothrow_move_constructible_v<_Start>)
      : view_interface<iota_view<_Start, _BoundSentinel>>()
      , __value_(::cuda::std::move(__value))
  {}

  _CCCL_API constexpr iota_view(
    type_identity_t<_Start> __value,
    type_identity_t<_BoundSentinel> __bound_sentinel) noexcept(is_nothrow_move_constructible_v<_Start>
                                                               && is_nothrow_move_constructible_v<_BoundSentinel>)
      : view_interface<iota_view<_Start, _BoundSentinel>>()
      , __value_(::cuda::std::move(__value))
      , __bound_sentinel_(::cuda::std::move(__bound_sentinel))
  {
    // Validate the precondition if possible.
    if constexpr (totally_ordered_with<_Start, _BoundSentinel>)
    {
      _CCCL_ASSERT(::cuda::std::ranges::less_equal()(__value_, __bound_sentinel_),
                   "Precondition violated: value is greater than bound.");
    }
  }

  _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
  _CCCL_REQUIRES(same_as<_Start, _BoundSentinel2>)
  _CCCL_API constexpr iota_view(__iterator __first, __iterator __last)
      : iota_view(::cuda::std::move(__first.__value_), ::cuda::std::move(__last.__value_))
  {}

  _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
  _CCCL_REQUIRES(same_as<_BoundSentinel2, unreachable_sentinel_t>)
  _CCCL_API constexpr iota_view(__iterator __first, _BoundSentinel __last)
      : iota_view(::cuda::std::move(__first.__value_), ::cuda::std::move(__last))
  {}

  _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
  _CCCL_REQUIRES((!same_as<_Start, _BoundSentinel2>) _CCCL_AND(!same_as<_Start, unreachable_sentinel_t>))
  _CCCL_API constexpr iota_view(__iterator __first, __sentinel __last)
      : iota_view(::cuda::std::move(__first.__value_), ::cuda::std::move(__last.__bound_sentinel_))
  {}

  [[nodiscard]] _CCCL_API constexpr __iterator begin() const
  {
    return __iterator{__value_};
  }

  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    if constexpr (same_as<_Start, _BoundSentinel>)
    {
      return __iterator{__bound_sentinel_};
    }
    else if constexpr (same_as<_BoundSentinel, unreachable_sentinel_t>)
    {
      return unreachable_sentinel;
    }
    else
    {
      return __sentinel{__bound_sentinel_};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _BoundSentinel2 = _BoundSentinel)
  _CCCL_REQUIRES((same_as<_Start, _BoundSentinel2> && __advanceable<_Start>)
                 || (integral<_Start> && integral<_BoundSentinel2>) || sized_sentinel_for<_BoundSentinel2, _Start>)
  _CCCL_API constexpr auto size() const
  {
    if constexpr (__integer_like<_Start> && __integer_like<_BoundSentinel>)
    {
      if (__value_ < 0)
      {
        if (__bound_sentinel_ < 0)
        {
          return ::cuda::std::__to_unsigned_like(-__value_) - ::cuda::std::__to_unsigned_like(-__bound_sentinel_);
        }
        return ::cuda::std::__to_unsigned_like(__bound_sentinel_) + ::cuda::std::__to_unsigned_like(-__value_);
      }
      return ::cuda::std::__to_unsigned_like(__bound_sentinel_) - ::cuda::std::__to_unsigned_like(__value_);
    }
    else
    {
      return ::cuda::std::__to_unsigned_like(__bound_sentinel_ - __value_);
    }
    _CCCL_UNREACHABLE();
  }
};

_CCCL_TEMPLATE(class _Start, class _BoundSentinel)
_CCCL_REQUIRES((!__integer_like<_Start> || !__integer_like<_BoundSentinel>
                || (__signed_integer_like<_Start> == __signed_integer_like<_BoundSentinel>) ))
_CCCL_HOST_DEVICE iota_view(_Start, _BoundSentinel) -> iota_view<_Start, _BoundSentinel>;

template <class _Start, class _BoundSentinel>
inline constexpr bool enable_borrowed_range<iota_view<_Start, _BoundSentinel>> = true;

_CCCL_END_NAMESPACE_RANGES

_CCCL_BEGIN_NAMESPACE_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__iota)

struct __fn
{
  template <class _Start>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Start&& __start) const
    noexcept(noexcept(::cuda::std::ranges::iota_view(::cuda::std::forward<_Start>(__start))))
      -> iota_view<remove_cvref_t<_Start>>
  {
    return ::cuda::std::ranges::iota_view(::cuda::std::forward<_Start>(__start));
  }

  template <class _Start, class _BoundSentinel>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Start&& __start, _BoundSentinel&& __bound_sentinel) const
    noexcept(noexcept(::cuda::std::ranges::iota_view(::cuda::std::forward<_Start>(__start),
                                                     ::cuda::std::forward<_BoundSentinel>(__bound_sentinel))))
      -> iota_view<remove_cvref_t<_Start>, remove_cvref_t<_BoundSentinel>>
  {
    return ::cuda::std::ranges::iota_view(
      ::cuda::std::forward<_Start>(__start), ::cuda::std::forward<_BoundSentinel>(__bound_sentinel));
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto iota = __iota::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_IOTA_VIEW_H
