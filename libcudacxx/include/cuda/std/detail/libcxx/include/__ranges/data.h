// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_DATA_H
#define _LIBCUDACXX___RANGES_DATA_H

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
#include "../__iterator/concepts.h"
#include "../__iterator/reverse_iterator.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/access.h"
#include "../__type_traits/is_pointer.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_pointer.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/auto_cast.h"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// [range.prim.data]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__data)

  template <class _Tp>
  _LIBCUDACXX_CONCEPT __ptr_to_object = is_pointer_v<_Tp> && is_object_v<remove_pointer_t<_Tp>>;

#if _CCCL_STD_VER >= 2020
  template <class _Tp>
  concept __member_data =
    __can_borrow<_Tp> &&
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      { _LIBCUDACXX_AUTO_CAST(__t.data()) } -> __ptr_to_object;
    };

  template <class _Tp>
  concept __ranges_begin_invocable =
    !__member_data<_Tp> &&
    __can_borrow<_Tp> &&
    requires(_Tp&& __t) {
      { _CUDA_VRANGES::begin(__t) } -> contiguous_iterator;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_data_,
    requires(_Tp&& __t)(
      requires(__can_borrow<_Tp>),
      requires(__workaround_52970<_Tp>),
      requires(__ptr_to_object<decltype(_LIBCUDACXX_AUTO_CAST(__t.data()))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_data = _LIBCUDACXX_FRAGMENT(__member_data_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __ranges_begin_invocable_,
    requires(_Tp&& __t)(
      requires(!__member_data<_Tp>),
      requires(__can_borrow<_Tp>),
      requires(contiguous_iterator<decltype(_CUDA_VRANGES::begin(__t))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __ranges_begin_invocable = _LIBCUDACXX_FRAGMENT(__ranges_begin_invocable_, _Tp);
#endif // _CCCL_STD_VER <= 2017

  struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Tp)
    _LIBCUDACXX_REQUIRES(__member_data<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(__t.data())) {
      return __t.data();
    }

  _LIBCUDACXX_TEMPLATE(class _Tp)
    _LIBCUDACXX_REQUIRES(__ranges_begin_invocable<_Tp>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(_CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__t)))) {
      return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__t));
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto data = __data::__fn{};
} // namespace __cpo

// [range.prim.cdata]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cdata)
  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_lvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(      _CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t)))
      { return          _CUDA_VRANGES::data(static_cast<const remove_reference_t<_Tp>&>(__t)); }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      _LIBCUDACXX_REQUIRES(is_rvalue_reference_v<_Tp&&>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::data(static_cast<const _Tp&&>(__t))))
      -> decltype(      _CUDA_VRANGES::data(static_cast<const _Tp&&>(__t)))
      { return          _CUDA_VRANGES::data(static_cast<const _Tp&&>(__t)); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto cdata = __cdata::__fn{};
} // namespace __cpo

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_DATA_H
