//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H
#define _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/__fwd/barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <>
class barrier<thread_scope_thread, ::cuda::std::__empty_completion> : private barrier<thread_scope_block>
{
  using __base = barrier<thread_scope_block>;

public:
  using __base::__base;

  _CCCL_API inline friend void init(barrier* __b,
                                    ::cuda::std::ptrdiff_t __expected,
                                    ::cuda::std::__empty_completion __completion = ::cuda::std::__empty_completion())
  {
    init(static_cast<__base*>(__b), __expected, __completion);
  }

  using __base::arrive;
  using __base::arrive_and_drop;
  using __base::arrive_and_wait;
  using __base::max;
  using __base::wait;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_BARRIER_THREAD_SCOPE_H
