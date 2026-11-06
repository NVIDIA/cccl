//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Cached pinned host staging for the combine-bearing algorithms.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/places.cuh>

#include <cstddef>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Cached thread-local pinned staging for small host combines.
//!
//! A cudaMallocHost/cudaFreeHost pair costs ~1 ms (OS page pinning) — orders
//! of magnitude more than the O(P) combine it would serve — so the default
//! staging is one thread-local pinned block, grown on demand and reused for
//! the thread's lifetime. Not reentrant within one thread (one combine at a
//! time uses it, which is how the algorithms call it). The destructor's
//! cudaFreeHost is best-effort: at thread/process teardown the runtime may
//! already be gone.
inline void* __pinned_staging(::std::size_t __bytes)
{
  struct __arena
  {
    void* __ptr          = nullptr;
    ::std::size_t __size = 0;
    ~__arena()
    {
      if (__ptr != nullptr)
      {
        (void) cudaFreeHost(__ptr);
      }
    }
  };
  static thread_local __arena __a;
  if (__a.__size < __bytes)
  {
    if (__a.__ptr != nullptr)
    {
      places::cuda_safe_call(cudaFreeHost(__a.__ptr));
    }
    void* __p = nullptr;
    places::cuda_safe_call(cudaMallocHost(&__p, __bytes));
    __a.__ptr  = __p;
    __a.__size = __bytes;
  }
  return __a.__ptr;
}
} // namespace reserved
} // namespace cuda::experimental::sharded
