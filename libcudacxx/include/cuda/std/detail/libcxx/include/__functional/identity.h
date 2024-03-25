// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_IDENTITY_H
#define _LIBCUDACXX___FUNCTIONAL_IDENTITY_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__functional/reference_wrapper.h"
#include "../__type_traits/integral_constant.h"
#include "../__utility/forward.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_identity : false_type
{};

struct __identity
{
  template <class _Tp>
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp&& operator()(_Tp&& __t) const noexcept
  {
    return _CUDA_VSTD::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

template <>
struct __is_identity<__identity> : true_type
{};
template <>
struct __is_identity<reference_wrapper<__identity> > : true_type
{};
template <>
struct __is_identity<reference_wrapper<const __identity> > : true_type
{};

#if _CCCL_STD_VER > 2011

struct identity
{
  template <class _Tp>
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp&& operator()(_Tp&& __t) const noexcept
  {
    return _CUDA_VSTD::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

template <>
struct __is_identity<identity> : true_type
{};
template <>
struct __is_identity<reference_wrapper<identity> > : true_type
{};
template <>
struct __is_identity<reference_wrapper<const identity> > : true_type
{};

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_IDENTITY_H
