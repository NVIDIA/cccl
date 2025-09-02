//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_REMOVE_H
#define _CUDA_STD___ALGORITHM_REMOVE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/find.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Tp>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
remove(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value_)
{
  __first = ::cuda::std::find(__first, __last, __value_);
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    while (++__i != __last)
    {
      if (!(*__i == __value_))
      {
        *__first = ::cuda::std::move(*__i);
        ++__first;
      }
    }
  }
  return __first;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_REMOVE_H
