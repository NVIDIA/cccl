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
class FirstCallGate
{
public:
  template <typename Invocation>
  int invoke(Invocation&& invocation)
  {
    if (complete_.load(std::memory_order_acquire))
    {
      return invocation();
    }

    std::unique_lock lock(mutex_);
    if (complete_.load(std::memory_order_relaxed))
    {
      lock.unlock();
      return invocation();
    }

    const int status = invocation();
    if (status == 0)
    {
      complete_.store(true, std::memory_order_release);
    }
    return status;
  }

private:
  std::atomic<bool> complete_{false};
  std::mutex mutex_;
};
} // namespace cccl::detail
