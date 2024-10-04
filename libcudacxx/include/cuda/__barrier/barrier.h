//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_BARRIER_H
#define _CUDA___BARRIER_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__new_>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <thread_scope _Sco, class _CompletionF>
class barrier : public _CUDA_VSTD::__barrier_base<_CompletionF, _Sco>
{
public:
  _CCCL_HIDE_FROM_ABI barrier() = default;

  barrier(const barrier&)            = delete;
  barrier& operator=(const barrier&) = delete;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr barrier(_CUDA_VSTD::ptrdiff_t __expected,
                                              _CompletionF __completion = _CompletionF())
      : _CUDA_VSTD::__barrier_base<_CompletionF, _Sco>(__expected, __completion)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI friend void init(barrier* __b, _CUDA_VSTD::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(__expected >= 0, "Cannot initialize barrier with negative arrival count");
    new (__b) barrier(__expected);
  }

  _LIBCUDACXX_HIDE_FROM_ABI friend void init(barrier* __b, _CUDA_VSTD::ptrdiff_t __expected, _CompletionF __completion)
  {
    _CCCL_ASSERT(__expected >= 0, "Cannot initialize barrier with negative arrival count");
    new (__b) barrier(__expected, __completion);
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_BARRIER_H
