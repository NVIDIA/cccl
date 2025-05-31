//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/execution.cuh>

#include <nv/target>

#include <cstdio>

#include <cuda_runtime_api.h>

namespace cudax = cuda::experimental;
namespace task  = cudax::execution;

// This example demonstrates how to use the experimental CUDA implementation of
// C++26's std::execution async tasking framework.

struct say_hello
{
  __device__ int operator()() const
  {
    printf("Hello from lambda on device!\n");
    return value;
  }

  int value;
};

__host__ void run()
{
  try
  {
    task::thread_context tctx;
    task::stream_context sctx;
    auto sch = sctx.get_scheduler();

    auto start = //
      task::schedule(sch) // begin work on the GPU
      | task::then(say_hello{42}) // enqueue a function object on the GPU
      | task::then([] __device__(int i) noexcept -> int { // enqueue a lambda on the GPU
          printf("Hello again from lambda on device! i = %d\n", i);
          return i + 1;
        })
      | task::continues_on(tctx.get_scheduler()) // continue work on the CPU
      | task::then([] __host__ __device__(int i) noexcept -> int { // run a lambda on the CPU
          NV_IF_TARGET(NV_IS_HOST,
                       (printf("Hello from lambda on host! i = %d\n", i);),
                       (printf("OOPS! still on the device! i = %d\n", i);))
          return i;
        });

    // run the task, wait for it to finish, and get the result
    auto [i] = task::sync_wait(std::move(start)).value();
    printf("All done on the host! result = %d\n", i);
  }
  catch (cuda::cuda_error const& e)
  {
    std::printf("CUDA error: %s\n", e.what());
  }
  catch (std::exception const& e)
  {
    std::printf("Exception: %s\n", e.what());
  }
  catch (...)
  {
    std::printf("Unknown exception\n");
  }
}

int main()
{
  run();
}
