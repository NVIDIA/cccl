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
namespace ex    = cudax::execution;

// This example demonstrates how to use the experimental CUDA implementation of
// C++26's std::execution async tasking framework.

int main()
{
  try
  {
    auto tctx = ex::thread_context{};
    auto sctx = ex::stream_context{cuda::device_ref{0}};
    auto gpu  = sctx.get_scheduler();

    const auto bulk_shape = 10;
    const auto bulk_fn    = [] __device__(const int index, int i) noexcept {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < bulk_shape)
      {
        printf("Hello from bulk task on device! index = %d, i = %d\n", index, i);
      }
    };

    auto start =
      // begin work on the GPU:
      ex::schedule(gpu)

      // execute a device lambda on the GPU:
      | ex::then([] __device__() noexcept -> int {
          printf("Hello from lambda on device!\n");
          return 42;
        })

      // do some parallel work on the GPU:
      | ex::bulk(ex::par, bulk_shape, bulk_fn) //

      // transfer execution back to the CPU:
      | ex::continues_on(tctx.get_scheduler())

      // execute a host/device lambda on the CPU:
      | ex::then([] __host__ __device__(int i) noexcept -> int {
          NV_IF_TARGET(NV_IS_HOST,
                       (printf("Hello from lambda on host! i = %d\n", i);),
                       (printf("OOPS! still on the device! i = %d\n", i);))
          return i + 1;
        });

    // run the task, wait for it to finish, and get the result
    auto [i] = ex::sync_wait(std::move(start)).value();
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
