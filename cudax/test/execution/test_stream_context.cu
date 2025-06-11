//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Include this first
#include <cuda/experimental/execution.cuh>

// Then include the test helpers
#include <nv/target>

#include "testing.cuh" // IWYU pragma: keep

_CCCL_NV_DIAG_SUPPRESS(177) // function "_is_on_device" was declared but never referenced

namespace
{
__host__ __device__ bool _is_on_device() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ return false; }),
                    ({ return true; }));
}

struct _say_hello
{
  __device__ int operator()() const
  {
    CUDAX_CHECK(_is_on_device());
    printf("Hello from lambda on device!\n");
    return value;
  }

  int value;
};

void stream_context_test1()
{
  cudax_async::stream_context ctx{cuda::experimental::device_ref{0}};
  auto sched = ctx.get_scheduler();

  auto sndr = cudax_async::schedule(sched) //
            | cudax_async::then([] __device__() noexcept -> bool {
                return _is_on_device();
              });

  auto [on_device] = cudax_async::sync_wait(std::move(sndr)).value();
  CHECK(on_device);
}

void stream_context_test2()
{
  cudax_async::thread_context tctx;
  cudax_async::stream_context sctx{cuda::experimental::device_ref{0}};
  auto sch = sctx.get_scheduler();

  auto start = //
    cudax_async::schedule(sch) // begin work on the GPU
    | cudax_async::then(_say_hello{42}) // enqueue a function object on the GPU
    | cudax_async::then([] __device__(int i) noexcept -> int { // enqueue a lambda on the GPU
        CUDAX_CHECK(_is_on_device());
        printf("Hello again from lambda on device! i = %d\n", i);
        return i + 1;
      })
    | cudax_async::continues_on(tctx.get_scheduler()) // continue work on the CPU
    | cudax_async::then([] __host__ __device__(int i) noexcept -> int { // run a lambda on the CPU
        CUDAX_CHECK(!_is_on_device());
        NV_IF_TARGET(NV_IS_HOST,
                     (printf("Hello from lambda on host! i = %d\n", i);),
                     (printf("OOPS! still on the device! i = %d\n", i);))
        return i;
      });

  // run the cudax_async, wait for it to finish, and get the result
  auto [i] = cudax_async::sync_wait(std::move(start)).value();
  CHECK(i == 43);
  printf("All done on the host! result = %d\n", i);
}

// Test code is placed in separate functions to avoid an nvc++ issue with
// extended lambdas in functions with internal linkage (as is the case
// with C2H tests).

C2H_TEST("a simple use of the stream context", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_context_test1());
}

C2H_TEST("a simple use of the stream context", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_context_test2());
}
} // namespace
