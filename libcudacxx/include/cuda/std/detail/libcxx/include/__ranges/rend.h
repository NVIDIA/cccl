// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_REND_H
#define _LIBCUDACXX___RANGES_REND_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/class_or_enum.h"
#include "../__concepts/same_as.h"
#include "../__iterator/concepts.h"
#include "../__iterator/readable_traits.h"
#include "../__iterator/reverse_iterator.h"
#include "../__ranges/access.h"
#include "../__ranges/rbegin.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/auto_cast.h"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// [range.access.rend]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__rend)
  template<class _Tp>
  void rend(_Tp&) = delete;
  template<class _Tp>
  void rend(const _Tp&) = delete;

#if _CCCL_STD_VER >= 2020
  template <class _Tp>
  concept __member_rend =
    __can_borrow<_Tp> &&
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      _CUDA_VRANGES::rbegin(__t);
      { _LIBCUDACXX_AUTO_CAST(__t.rend()) } -> sentinel_for<decltype(_CUDA_VRANGES::rbegin(__t))>;
    };


  template <class _Tp>
  concept __unqualified_rend =
    !__member_rend<_Tp> &&
    __can_borrow<_Tp> &&
    __class_or_enum<remove_cvref_t<_Tp>> &&
    requires(_Tp&& __t) {
      _CUDA_VRANGES::rbegin(__t);
      { _LIBCUDACXX_AUTO_CAST(rend(__t)) } -> sentinel_for<decltype(_CUDA_VRANGES::rbegin(__t))>;
    };

  template <class _Tp>
  concept __can_reverse =
    __can_borrow<_Tp> &&
    !__member_rend<_Tp> &&
    !__unqualified_rend<_Tp> &&
    requires(_Tp&& __t) {
      { _CUDA_VRANGES::begin(__t) } -> same_as<decltype(_CUDA_VRANGES::end(__t))>;
      { _CUDA_VRANGES::begin(__t) } -> bidirectional_iterator;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_rend_,
    requires(_Tp&& __t)(
      requires(__can_borrow<_Tp>),
      requires(__workaround_52970<_Tp>),
      typename(decltype(_CUDA_VRANGES::rbegin(__t))),
      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(__t.rend())), decltype(_CUDA_VRANGES::rbegin(__t))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_rend = _LIBCUDACXX_FRAGMENT(__member_rend_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_rend_,
    requires(_Tp&& __t)(
      requires(!__member_rend<_Tp>),
      requires(__can_borrow<_Tp>),
      requires(__class_or_enum<remove_cvref_t<_Tp>>),
      typename(decltype(_CUDA_VRANGES::rbegin(__t))),
      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(rend(__t))), decltype(_CUDA_VRANGES::rbegin(__t))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __unqualified_rend = _LIBCUDACXX_FRAGMENT(__unqualified_rend_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_reverse_,
    requires(_Tp&& __t)(
      requires(!__member_rend<_Tp>),
      requires(!__unqualified_rend<_Tp>),
      requires(__can_borrow<_Tp>),
      requires(same_as<decltype(_CUDA_VRANGES::begin(__t)), decltype(_CUDA_VRANGES::end(__t))>),
      requires(bidirectional_iterator<decltype(_CUDA_VRANGES::begin(__t))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_reverse = _LIBCUDACXX_FRAGMENT(__can_reverse_, _Tp);
#endif // _CCCL_STD_VER <= 2017

  class __fn {
  public:
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__member_rend<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.rend())))
    {
      return _LIBCUDACXX_AUTO_CAST(__t.rend());
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__unqualified_rend<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(rend(__t))))
    {
      return _LIBCUDACXX_AUTO_CAST(rend(__t));
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(__can_reverse<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::begin(__t)))
    {
      return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::begin(__t));
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES((!__member_rend<_Tp> && !__unqualified_rend<_Tp> && !__can_reverse<_Tp>))
    void operator()(_Tp&&) const = delete;
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto rend = __rend::__fn{};
} // namespace __cpo

// [range.access.crend]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__crend)
  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_lvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::rend(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(      _CUDA_VRANGES::rend(static_cast<const remove_reference_t<_Tp>&>(__t)))
      { return          _CUDA_VRANGES::rend(static_cast<const remove_reference_t<_Tp>&>(__t)); }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_rvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::rend(static_cast<const _Tp&&>(__t))))
      -> decltype(      _CUDA_VRANGES::rend(static_cast<const _Tp&&>(__t)))
      { return          _CUDA_VRANGES::rend(static_cast<const _Tp&&>(__t)); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto crend = __crend::__fn{};
} // namespace __cpo

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_REND_H
