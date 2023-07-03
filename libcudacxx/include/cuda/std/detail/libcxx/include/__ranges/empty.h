// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_EMPTY_H
#define _LIBCUDACXX___RANGES_EMPTY_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/class_or_enum.h"
#include "../__iterator/concepts.h"
#include "../__ranges/access.h"
#include "../__ranges/size.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

// [range.prim.empty]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__empty)

#if _LIBCUDACXX_STD_VER > 17
  template <class _Tp>
  concept __member_empty =
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      bool(__t.empty());
    };

  template<class _Tp>
  concept __can_invoke_size =
    !__member_empty<_Tp> &&
    requires(_Tp&& __t) { _CUDA_VRANGES::size(__t); };

  template <class _Tp>
  concept __can_compare_begin_end =
    !__member_empty<_Tp> &&
    !__can_invoke_size<_Tp> &&
    requires(_Tp&& __t) {
      bool(_CUDA_VRANGES::begin(__t) == _CUDA_VRANGES::end(__t));
      { _CUDA_VRANGES::begin(__t) } -> forward_iterator;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_empty_,
    requires(_Tp&& __t)(
      requires(__workaround_52970<_Tp>),
      (bool(__t.empty()))
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_empty = _LIBCUDACXX_FRAGMENT(__member_empty_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_invoke_size_,
    requires(_Tp&& __t)(
      requires(!__member_empty<_Tp>),
      ((void) _CUDA_VRANGES::size(__t))
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_invoke_size = _LIBCUDACXX_FRAGMENT(__can_invoke_size_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_compare_begin_end_,
    requires(_Tp&& __t)(
      requires(!__member_empty<_Tp>),
      requires(!__can_invoke_size<_Tp>),
      (bool(_CUDA_VRANGES::begin(__t) == _CUDA_VRANGES::end(__t))),
      requires(forward_iterator<decltype(_CUDA_VRANGES::begin(__t))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_compare_begin_end = _LIBCUDACXX_FRAGMENT(__can_compare_begin_end_, _Tp);
#endif // _LIBCUDACXX_STD_VER < 20

  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __member_empty<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr bool operator()(_Tp&& __t) const
      noexcept(noexcept(bool(__t.empty()))) {
      return bool(__t.empty());
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __can_invoke_size<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr bool operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::size(__t))) {
      return _CUDA_VRANGES::size(__t) == 0;
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __can_compare_begin_end<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr bool operator()(_Tp&& __t) const
      noexcept(noexcept(bool(_CUDA_VRANGES::begin(__t) == _CUDA_VRANGES::end(__t)))) {
      return _CUDA_VRANGES::begin(__t) == _CUDA_VRANGES::end(__t);
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto empty = __empty::__fn{};
} // namespace __cpo

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_EMPTY_H
