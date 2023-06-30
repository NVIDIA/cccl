// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ITER_MOVE_H
#define _LIBCUDACXX___ITERATOR_ITER_MOVE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/class_or_enum.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvoid-ptr-dereference"
#endif


#if _LIBCUDACXX_STD_VER > 14

// [iterator.cust.move]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__iter_move)

_LIBCUDACXX_INLINE_VISIBILITY
void iter_move();

#if LIBCUDACXX_STD_VER > 17
template <class _Tp>
concept __unqualified_iter_move =
  __class_or_enum<remove_cvref_t<_Tp>> &&
  requires (_Tp&& __t) {
    iter_move(_CUDA_VSTD::forward<_Tp>(__t));
  };

template<class _Tp>
concept __move_deref =
  !__unqualified_iter_move<_Tp> &&
  requires (_Tp&& __t) {
    *__t;
    requires is_lvalue_reference_v<decltype(*__t)>;
  };

template<class _Tp>
concept __just_deref =
  !__unqualified_iter_move<_Tp> &&
  !__move_deref<_Tp> &&
  requires (_Tp&& __t) {
    *__t;
    requires (!is_lvalue_reference_v<decltype(*__t)>);
  };

#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __unqualified_iter_move_,
  requires(_Tp&& __t)(
    requires(__class_or_enum<remove_cvref_t<_Tp>>),
    (iter_move(_CUDA_VSTD::forward<_Tp>(__t)))
  ));

template <class _Tp>
_LIBCUDACXX_CONCEPT __unqualified_iter_move = _LIBCUDACXX_FRAGMENT(__unqualified_iter_move_, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __move_deref_,
  requires(_Tp&& __t)(
    requires(!__unqualified_iter_move<_Tp>),
    requires(is_lvalue_reference_v<decltype(*__t)>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT __move_deref = _LIBCUDACXX_FRAGMENT(__move_deref_, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __just_deref_,
  requires(_Tp&& __t)(
    requires(!__unqualified_iter_move<_Tp>),
    requires(!__move_deref<_Tp>),
    requires(!is_lvalue_reference_v<decltype(*__t)>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT __just_deref = _LIBCUDACXX_FRAGMENT(__just_deref_, _Tp);
#endif // LIBCUDACXX_STD_VER < 20

// [iterator.cust.move]

struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Ip)
    (requires __unqualified_iter_move<_Ip>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator()(_Ip&& __i) const
    noexcept(noexcept(iter_move(_CUDA_VSTD::forward<_Ip>(__i))))
  {
    return iter_move(_CUDA_VSTD::forward<_Ip>(__i));
  }

  _LIBCUDACXX_TEMPLATE(class _Ip)
    (requires __move_deref<_Ip>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Ip&& __i) const
    noexcept(noexcept(_CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i))))
    -> decltype(      _CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i)))
    { return          _CUDA_VSTD::move(*_CUDA_VSTD::forward<_Ip>(__i)); }

  _LIBCUDACXX_TEMPLATE(class _Ip)
    (requires __just_deref<_Ip>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Ip&& __i) const
    noexcept(noexcept(*_CUDA_VSTD::forward<_Ip>(__i)))
    -> decltype(      *_CUDA_VSTD::forward<_Ip>(__i))
    { return          *_CUDA_VSTD::forward<_Ip>(__i); }
};
_LIBCUDACXX_END_NAMESPACE_CPO
inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto iter_move = __iter_move::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17
template<__dereferenceable _Tp>
  requires requires(_Tp& __t) { { _CUDA_VRANGES::iter_move(__t) } -> __can_reference; }
using iter_rvalue_reference_t = decltype(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Tp&>()));

#else

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __can_iter_rvalue_reference_t_,
  requires(_Tp& __t)(
    requires(__dereferenceable<_Tp>),
    requires(__can_reference<decltype(_CUDA_VRANGES::iter_move(__t))>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT __can_iter_rvalue_reference_t =  _LIBCUDACXX_FRAGMENT(__can_iter_rvalue_reference_t_, _Tp);

template<class _Tp>
using __iter_rvalue_reference_t = decltype(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<_Tp&>()));

template<class _Tp>
using iter_rvalue_reference_t = enable_if_t<__can_iter_rvalue_reference_t<_Tp>,
                                            __iter_rvalue_reference_t<_Tp>>;
#endif // LIBCUDACXX_STD_VER < 20

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 14

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif // _LIBCUDACXX___ITERATOR_ITER_MOVE_H
