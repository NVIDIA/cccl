//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "config.cuh"

#if !defined(__CUDA_ARCH__)

#  include <thread>

#  include "run_loop.cuh"

// Must be the last include
#  include "prologue.cuh"

namespace cuda::experimental::__async
{
struct thread_context
{
  thread_context() noexcept
      : _thread{[this] {
        _loop.run();
      }}
  {}

  ~thread_context() noexcept
  {
    join();
  }

  void join() noexcept
  {
    if (_thread.joinable())
    {
      _loop.finish();
      _thread.join();
    }
  }

  auto get_scheduler()
  {
    return _loop.get_scheduler();
  }

private:
  run_loop _loop;
  ::std::thread _thread;
};
} // namespace cuda::experimental::__async

#  include "epilogue.cuh"

#endif // !defined(__CUDA_ARCH__)
