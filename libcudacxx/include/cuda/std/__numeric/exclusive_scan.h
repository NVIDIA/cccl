// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_EXCLUSIVE_SCAN_H
#define _CUDA_STD___NUMERIC_EXCLUSIVE_SCAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
_CCCL_API constexpr _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init, _BinaryOp __b)
{
  if (__first != __last)
  {
    _Tp __tmp(__b(__init, *__first));
    while (true)
    {
      *__result = ::cuda::std::move(__init);
      ++__result;
      ++__first;
      if (__first == __last)
      {
        break;
      }
      __init = ::cuda::std::move(__tmp);
      __tmp  = __b(__init, *__first);
    }
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _Tp>
_CCCL_API constexpr _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init)
{
  return ::cuda::std::exclusive_scan(__first, __last, __result, __init, ::cuda::std::plus<>());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_EXCLUSIVE_SCAN_H
