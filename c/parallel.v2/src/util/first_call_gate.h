//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <mutex>

namespace cccl::detail
{
// Serializes the first successful invocation of generated CUB code that
// lazily initializes a function-local static. Windows HostJIT translation
// units use -fno-threadsafe-statics because the required CRT support is not
// available. Callers bypass the gate for empty work; after initialization,
// the atomic fast path invokes the generated function without locking.
class first_call_gate
{
public:
  template <typename Invocation>
  int invoke(Invocation&& invocation)
  {
    if (complete.load(std::memory_order_acquire))
    {
      return invocation();
    }

    std::unique_lock lock(mutex);
    if (complete.load(std::memory_order_relaxed))
    {
      lock.unlock();
      return invocation();
    }

    const int status = invocation();
    if (status == 0)
    {
      complete.store(true, std::memory_order_release);
    }
    return status;
  }

private:
  std::atomic<bool> complete{false};
  std::mutex mutex;
};
} // namespace cccl::detail
