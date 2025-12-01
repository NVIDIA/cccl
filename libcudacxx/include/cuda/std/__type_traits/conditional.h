//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H
#define _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <bool _Bp, class _Tp, class _Fp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional
{
  using type = _Tp;
};
template <class _Tp, class _Fp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional<false, _Tp, _Fp>
{
  using type = _Fp;
};

template <bool>
struct __conditional_t_impl
{
  template <class _Tp, class _Fp>
  using _Select _CCCL_NODEBUG_ALIAS = _Tp;
};
template <>
struct __conditional_t_impl<false>
{
  template <class _Tp, class _Fp>
  using _Select _CCCL_NODEBUG_ALIAS = _Fp;
};

template <bool _Bp, class _Tp, class _Fp>
using conditional_t _CCCL_NODEBUG_ALIAS = typename __conditional_t_impl<_Bp>::template _Select<_Tp, _Fp>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_CONDITIONAL_H
