// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___ITERATOR_ITER_SWAP_H
#define _CUDA_STD___ITERATOR_ITER_SWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__concepts/swappable.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

// [iter.cust.swap]
_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES
_CCCL_BEGIN_NAMESPACE_CPO(__iter_swap)
template <class _I1, class _I2>
void iter_swap(_I1, _I2) = delete;

template <class _T1, class _T2>
_CCCL_CONCEPT __unqualified_iter_swap = _CCCL_REQUIRES_EXPR((_T1, _T2), _T1&& __x, _T2&& __y)(
  requires(__class_or_enum<remove_cvref_t<_T1>> || __class_or_enum<remove_cvref_t<_T2>>),
  ((void) iter_swap(::cuda::std::forward<_T1>(__x), ::cuda::std::forward<_T2>(__y))));

#if _CCCL_HAS_NOEXCEPT_MANGLING() // older GCC cannot use noexcept inside a requires clause
template <class _T1, class _T2>
_CCCL_CONCEPT __noexcept_unqualified_iter_swap = _CCCL_REQUIRES_EXPR((_T1, _T2), _T1&& __x, _T2&& __y)(
  requires(__unqualified_iter_swap<_T1, _T2>),
  noexcept(iter_swap(::cuda::std::forward<_T1>(__x), ::cuda::std::forward<_T2>(__y))));
#else // ^^^ _CCCL_HAS_NOEXCEPT_MANGLING() ^^^ / vvv !_CCCL_HAS_NOEXCEPT_MANGLING() vvv
template <class _T1, class _T2, bool = __unqualified_iter_swap<_T1, _T2>>
inline constexpr bool __noexcept_unqualified_iter_swap = false;

template <class _T1, class _T2>
inline constexpr bool __noexcept_unqualified_iter_swap<_T1, _T2, true> =
  noexcept(iter_swap(::cuda::std::declval<_T1>(), ::cuda::std::declval<_T2>()));
#endif // !_CCCL_HAS_NOEXCEPT_MANGLING()

template <class _T1, class _T2>
_CCCL_CONCEPT __readable_swappable = _CCCL_REQUIRES_EXPR((_T1, _T2))(
  requires(!__unqualified_iter_swap<_T1, _T2>),
  requires(indirectly_readable<_T1>),
  requires(indirectly_readable<_T2>),
  requires(__can_reference<iter_reference_t<_T1>>),
  requires(__can_reference<iter_reference_t<_T2>>),
  requires(swappable_with<iter_reference_t<_T1>, iter_reference_t<_T2>>));

#if _CCCL_HAS_NOEXCEPT_MANGLING() // older GCC cannot use noexcept inside a requires clause
template <class _T1, class _T2>
_CCCL_CONCEPT __noexcept_readable_swappable = _CCCL_REQUIRES_EXPR((_T1, _T2), _T1&& __x, _T2&& __y) //
  (requires(__readable_swappable<_T1, _T2>),
   noexcept(::cuda::std::ranges::swap(*::cuda::std::forward<_T1>(__x), *::cuda::std::forward<_T2>(__y))));
#else // ^^^ _CCCL_HAS_NOEXCEPT_MANGLING() ^^^ / vvv !_CCCL_HAS_NOEXCEPT_MANGLING() vvv
template <class _T1, class _T2, bool = __readable_swappable<_T1, _T2>>
inline constexpr bool __noexcept_readable_swappable = false;

template <class _T1, class _T2>
inline constexpr bool __noexcept_readable_swappable<_T1, _T2, true> =
  noexcept(::cuda::std::ranges::swap(*::cuda::std::declval<_T1>(), *::cuda::std::declval<_T2>()));
#endif // !_CCCL_HAS_NOEXCEPT_MANGLING()

template <class _T1, class _T2>
_CCCL_CONCEPT __movable_storable = _CCCL_REQUIRES_EXPR((_T1, _T2))(
  requires(!__unqualified_iter_swap<_T1, _T2>),
  requires(!__readable_swappable<_T1, _T2>),
  requires(indirectly_movable_storable<_T1, _T2>),
  requires(indirectly_movable_storable<_T2, _T1>));

#if _CCCL_HAS_NOEXCEPT_MANGLING() // older GCC cannot use noexcept inside a requires clause
template <class _T1, class _T2>
_CCCL_CONCEPT __noexcept_movable_storable =
  _CCCL_REQUIRES_EXPR((_T1, _T2), _T1&& __x, _T2&& __y, iter_value_t<_T2> __old)(
    requires(__movable_storable<_T1, _T2>),
    noexcept(iter_value_t<_T2>(::cuda::std::ranges::iter_move(__y))),
    noexcept(*__y = ::cuda::std::ranges::iter_move(__x)),
    noexcept(*::cuda::std::forward<_T1>(__x) = ::cuda::std::move(__old)));
#else // ^^^ _CCCL_HAS_NOEXCEPT_MANGLING() ^^^ / vvv !_CCCL_HAS_NOEXCEPT_MANGLING() vvv
template <class _T1, class _T2, bool = __movable_storable<_T1, _T2>>
inline constexpr bool __noexcept_movable_storable = false;

template <class _T1, class _T2>
inline constexpr bool __noexcept_movable_storable<_T1, _T2, true> =
  noexcept(iter_value_t<_T2>(::cuda::std::ranges::iter_move(::cuda::std::declval<add_lvalue_reference_t<_T2>>())))
  && noexcept(*::cuda::std::declval<add_lvalue_reference_t<_T2>>() =
                ::cuda::std::ranges::iter_move(::cuda::std::declval<add_lvalue_reference_t<_T1>>()))
  && noexcept(*::cuda::std::declval<_T1>() = ::cuda::std::declval<iter_value_t<_T2>>());
#endif // !_CCCL_HAS_NOEXCEPT_MANGLING()

struct __fn
{
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(__unqualified_iter_swap<_T1, _T2>)
  _CCCL_API constexpr void operator()(_T1&& __x, _T2&& __y) const noexcept(__noexcept_unqualified_iter_swap<_T1, _T2>)
  {
    (void) iter_swap(::cuda::std::forward<_T1>(__x), ::cuda::std::forward<_T2>(__y));
  }

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(__readable_swappable<_T1, _T2>)
  _CCCL_API constexpr void operator()(_T1&& __x, _T2&& __y) const noexcept(__noexcept_readable_swappable<_T1, _T2>)
  {
    ::cuda::std::ranges::swap(*::cuda::std::forward<_T1>(__x), *::cuda::std::forward<_T2>(__y));
  }

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(__movable_storable<_T2, _T1>)
  _CCCL_API constexpr void operator()(_T1&& __x, _T2&& __y) const noexcept(__noexcept_movable_storable<_T1, _T2>)
  {
    iter_value_t<_T2> __old(::cuda::std::ranges::iter_move(__y));
    *__y                            = ::cuda::std::ranges::iter_move(__x);
    *::cuda::std::forward<_T1>(__x) = ::cuda::std::move(__old);
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto iter_swap = __iter_swap::__fn{};
} // namespace __cpo
_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD
#if _CCCL_HAS_CONCEPTS()
template <class _I1, class _I2 = _I1>
concept indirectly_swappable =
  indirectly_readable<_I1> && indirectly_readable<_I2> && requires(const _I1 __i1, const _I2 __i2) {
    ::cuda::std::ranges::iter_swap(__i1, __i1);
    ::cuda::std::ranges::iter_swap(__i2, __i2);
    ::cuda::std::ranges::iter_swap(__i1, __i2);
    ::cuda::std::ranges::iter_swap(__i2, __i1);
  };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv _CCCL_HAS_CONCEPTS() vvv
template <class _I1, class _I2>
_CCCL_CONCEPT_FRAGMENT(
  __indirectly_swappable_,
  requires(const _I1 __i1, const _I2 __i2)(
    requires(indirectly_readable<_I1>),
    requires(indirectly_readable<_I2>),
    (::cuda::std::ranges::iter_swap(__i1, __i1)),
    (::cuda::std::ranges::iter_swap(__i2, __i2)),
    (::cuda::std::ranges::iter_swap(__i1, __i2)),
    (::cuda::std::ranges::iter_swap(__i2, __i1))));

template <class _I1, class _I2 = _I1>
_CCCL_CONCEPT indirectly_swappable = _CCCL_FRAGMENT(__indirectly_swappable_, _I1, _I2);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

template <class _I1, class _I2 = _I1, class = void>
inline constexpr bool __noexcept_swappable = false;

template <class _I1, class _I2>
inline constexpr bool __noexcept_swappable<_I1, _I2, enable_if_t<indirectly_swappable<_I1, _I2>>> =
  noexcept(::cuda::std::ranges::iter_swap(::cuda::std::declval<_I1&>(), ::cuda::std::declval<_I2&>()));

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_ITER_SWAP_H
