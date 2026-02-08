//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_THREAD_CONTEXT
#define __CUDAX_EXECUTION_THREAD_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/run_loop.cuh>

#include <thread>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT thread_context
{
  _CCCL_HOST_API thread_context() noexcept
      : __thrd_{[this] {
        __loop_.run();
      }}
  {}

  _CCCL_HOST_API ~thread_context() noexcept
  {
    join();
  }

  _CCCL_HOST_API void join() noexcept
  {
    if (__thrd_.joinable())
    {
      __loop_.finish();
      __thrd_.join();
    }
  }

  _CCCL_API auto get_scheduler()
  {
    return __loop_.get_scheduler();
  }

  _CCCL_HOST_API auto get_id() const noexcept
  {
    return __thrd_.get_id();
  }

private:
  run_loop __loop_;
  ::std::thread __thrd_;
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_THREAD_CONTEXT
