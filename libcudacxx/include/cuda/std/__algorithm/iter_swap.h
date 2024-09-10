//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_ITER_SWAP_H
#define _LIBCUDACXX___ALGORITHM_ITER_SWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/swap.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 void iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) noexcept(
  noexcept(swap(*_CUDA_VSTD::declval<_ForwardIterator1>(), *_CUDA_VSTD::declval<_ForwardIterator2>())))
{
  swap(*__a, *__b);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_ITER_SWAP_H
