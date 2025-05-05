//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_THREAD_CONTEXT
#define __CUDAX_ASYNC_DETAIL_THREAD_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__detail/config.cuh>

#if !defined(__CUDA_ARCH__)

#  include <cuda/experimental/__async/sender/run_loop.cuh>

#  include <thread>

#  include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT thread_context
{
  thread_context() noexcept
      : __thrd_{[this] {
        __loop_.run();
      }}
  {}

  ~thread_context() noexcept
  {
    join();
  }

  void join() noexcept
  {
    if (__thrd_.joinable())
    {
      __loop_.finish();
      __thrd_.join();
    }
  }

  auto get_scheduler()
  {
    return __loop_.get_scheduler();
  }

private:
  run_loop __loop_;
  ::std::thread __thrd_;
};
} // namespace cuda::experimental::__async

#  include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // !defined(__CUDA_ARCH__)

#endif
