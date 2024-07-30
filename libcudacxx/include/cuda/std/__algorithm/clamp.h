//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_CLAMP_H
#define _LIBCUDACXX___ALGORITHM_CLAMP_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/detail/libcxx/include/__assert>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class _Compare>
_CCCL_NODISCARD inline _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi, _Compare __comp)
{
  _LIBCUDACXX_ASSERT(!__comp(__hi, __lo), "Bad bounds passed to std::clamp");
  return __comp(__v, __lo) ? __lo : __comp(__hi, __v) ? __hi : __v;
}

template <class _Tp>
_CCCL_NODISCARD inline _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi)
{
  return _CUDA_VSTD::clamp(__v, __lo, __hi, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_CLAMP_H
