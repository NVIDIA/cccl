// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___ITERATOR_ITER_SWAP_H
#define _LIBCUDACXX___ITERATOR_ITER_SWAP_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/class_or_enum.h"
#include "../__concepts/swappable.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iter_move.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/readable_traits.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

// [iter.cust.swap]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__iter_swap)
  template<class _I1, class _I2>
  void iter_swap(_I1, _I2) = delete;

#if _LIBCUDACXX_STD_VER > 17
  template<class _T1, class _T2>
  concept __unqualified_iter_swap =
    (__class_or_enum<remove_cvref_t<_T1>> || __class_or_enum<remove_cvref_t<_T2>>) &&
    requires (_T1&& __x, _T2&& __y) {
      iter_swap(_CUDA_VSTD::forward<_T1>(__x), _CUDA_VSTD::forward<_T2>(__y));
    };

  template<class _T1, class _T2>
  concept __readable_swappable = !__unqualified_iter_swap<_T1, _T2>
                              && indirectly_readable<_T1>
                              && indirectly_readable<_T2>
                              && swappable_with<iter_reference_t<_T1>, iter_reference_t<_T2>>;

  template<class _T1, class _T2>
  concept __moveable_storable = !__unqualified_iter_swap<_T1, _T2>
                             && !__readable_swappable<_T1, _T2>
                             && indirectly_movable_storable<_T1, _T2>
                             && indirectly_movable_storable<_T2, _T1>;
#else
  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_iter_swap_,
    requires(_T1&& __x, _T2&& __y)(
      requires(__class_or_enum<remove_cvref_t<_T1>> || __class_or_enum<remove_cvref_t<_T2>>),
      ((void) iter_swap(_CUDA_VSTD::forward<_T1>(__x), _CUDA_VSTD::forward<_T2>(__y)))
    ));

  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT __unqualified_iter_swap = _LIBCUDACXX_FRAGMENT(__unqualified_iter_swap_, _T1, _T2);

  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __readable_swappable_,
    requires()(
      requires(!__unqualified_iter_swap<_T1, _T2>),
      requires(indirectly_readable<_T1>),
      requires(indirectly_readable<_T2>),
      requires(swappable_with<iter_reference_t<_T1>, iter_reference_t<_T2>>)
    ));

  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT __readable_swappable = _LIBCUDACXX_FRAGMENT(__readable_swappable_, _T1, _T2);

  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __moveable_storable_,
    requires()(
      requires(!__unqualified_iter_swap<_T1, _T2>),
      requires(!__readable_swappable<_T1, _T2>),
      requires(indirectly_movable_storable<_T1, _T2>),
      requires(indirectly_movable_storable<_T2, _T1>)
    ));

  template<class _T1, class _T2>
  _LIBCUDACXX_CONCEPT __moveable_storable = _LIBCUDACXX_FRAGMENT(__moveable_storable_, _T1, _T2);
#endif // _LIBCUDACXX_STD_VER > 11

  struct __fn {
  _LIBCUDACXX_TEMPLATE(class _T1, class _T2)
    (requires __unqualified_iter_swap<_T1, _T2>)
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(iter_swap(_CUDA_VSTD::forward<_T1>(__x), _CUDA_VSTD::forward<_T2>(__y))))
    {
      (void)iter_swap(_CUDA_VSTD::forward<_T1>(__x), _CUDA_VSTD::forward<_T2>(__y));
    }

  _LIBCUDACXX_TEMPLATE(class _T1, class _T2)
    (requires __readable_swappable<_T1, _T2>)
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(_CUDA_VRANGES::swap(*_CUDA_VSTD::forward<_T1>(__x), *_CUDA_VSTD::forward<_T2>(__y))))
    {
      _CUDA_VRANGES::swap(*_CUDA_VSTD::forward<_T1>(__x), *_CUDA_VSTD::forward<_T2>(__y));
    }

  _LIBCUDACXX_TEMPLATE(class _T1, class _T2)
    (requires __moveable_storable<_T2, _T1>)
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(iter_value_t<_T2>(_CUDA_VRANGES::iter_move(__y))) &&
               noexcept(*__y = _CUDA_VRANGES::iter_move(__x)) &&
               noexcept(*_CUDA_VSTD::forward<_T1>(__x) = declval<iter_value_t<_T2>>()))
    {
      iter_value_t<_T2> __old(_CUDA_VRANGES::iter_move(__y));
      *__y = _CUDA_VRANGES::iter_move(__x);
      *_CUDA_VSTD::forward<_T1>(__x) = _CUDA_VSTD::move(__old);
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto iter_swap = __iter_swap::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD
#if _LIBCUDACXX_STD_VER > 17
template<class _I1, class _I2 = _I1>
concept indirectly_swappable =
  indirectly_readable<_I1> && indirectly_readable<_I2> &&
  requires(const _I1 __i1, const _I2 __i2) {
    _CUDA_VRANGES::iter_swap(__i1, __i1);
    _CUDA_VRANGES::iter_swap(__i2, __i2);
    _CUDA_VRANGES::iter_swap(__i1, __i2);
    _CUDA_VRANGES::iter_swap(__i2, __i1);
  };
#else
template<class _I1, class _I2>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __indirectly_swappable_,
  requires(const _I1 __i1, const _I2 __i2)(
    requires(indirectly_readable<_I1>),
    requires(indirectly_readable<_I2>),
    (_CUDA_VRANGES::iter_swap(__i1, __i1)),
    (_CUDA_VRANGES::iter_swap(__i2, __i2)),
    (_CUDA_VRANGES::iter_swap(__i1, __i2)),
    (_CUDA_VRANGES::iter_swap(__i2, __i1))
  ));

template<class _I1, class _I2 = _I1>
_LIBCUDACXX_CONCEPT indirectly_swappable = _LIBCUDACXX_FRAGMENT(__indirectly_swappable_, _I1, _I2);
#endif // _LIBCUDACXX_STD_VER > 17

template<class _I1, class _I2 = _I1, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_swappable = false;

template<class _I1, class _I2>
_LIBCUDACXX_INLINE_VAR constexpr bool __noexcept_swappable<_I1, _I2, __enable_if_t<indirectly_swappable<_I1, _I2>>>
  = noexcept(_CUDA_VRANGES::iter_swap(_CUDA_VSTD::declval<_I1&>(), _CUDA_VSTD::declval<_I2&>()));

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___ITERATOR_ITER_SWAP_H
