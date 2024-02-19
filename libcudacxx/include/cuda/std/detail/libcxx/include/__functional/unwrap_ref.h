//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_UNWRAP_REF_H
#define _LIBCUDACXX___FUNCTIONAL_UNWRAP_REF_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __unwrap_reference { typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type; };

template <class _Tp>
class reference_wrapper;

template <class _Tp>
struct __unwrap_reference<reference_wrapper<_Tp> > { typedef _LIBCUDACXX_NODEBUG_TYPE _Tp& type; };

template <class _Tp>
struct decay;

#if _CCCL_STD_VER > 2017
template <class _Tp>
struct unwrap_reference : __unwrap_reference<_Tp> { };

template <class _Tp>
using unwrap_reference_t = typename unwrap_reference<_Tp>::type;

template <class _Tp>
struct unwrap_ref_decay : unwrap_reference<typename decay<_Tp>::type> { };

template <class _Tp>
using unwrap_ref_decay_t = typename unwrap_ref_decay<_Tp>::type;
#endif // _CCCL_STD_VER > 2017

template <class _Tp>
struct __unwrap_ref_decay
#if _CCCL_STD_VER > 2017
    : unwrap_ref_decay<_Tp>
#else
    : __unwrap_reference<typename decay<_Tp>::type>
#endif
{ };

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_UNWRAP_REF_H
