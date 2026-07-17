//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/memory_pool>

#include <testing.cuh>

// This test is an exception and shouldn't use C2H_CCCLRT_TEST macro
C2H_TEST("Call each driver api", "[utility]")
{
  namespace driver = ::cuda::__driver;
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  CUDART(cudaStreamCreate(&stream));

  auto ctx = driver::__ctxGetCurrent();
  CCCLRT_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  // Confirm we can push multiple times
  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPush(ctx);
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  // Confirm stream ctx match
  auto stream_ctx = driver::__streamGetCtx(stream);
  CCCLRT_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CCCLRT_REQUIRE(driver::__deviceGet(0) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = driver::__primaryCtxRetain(0);
  CCCLRT_REQUIRE(ctx == primary_ctx);

  driver::__ctxPop();
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == nullptr);

  CCCLRT_REQUIRE(driver::__isPrimaryCtxActive(0));
  // Confirm we can reset the primary context with double release
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);
  CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);

  // Try a third release in case curand retained the primary ctx as well
  if (driver::__isPrimaryCtxActive(0))
  {
    CCCLRT_REQUIRE(driver::__primaryCtxReleaseNoThrow(0) == cudaSuccess);
  }

  CCCLRT_REQUIRE(!driver::__isPrimaryCtxActive(0));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  CUDART(driver::__streamDestroyNoThrow(stream));
}

C2H_CCCLRT_TEST("memcpy async uses the stream context", "[utility][multi_gpu]")
{
  namespace driver = ::cuda::__driver;

  if (cuda::devices.size() < 2)
  {
    SKIP("Need at least 2 devices");
  }

  cuda::device_ref current_device{0};
  cuda::device_ref stream_device{1};
  const auto initial_stack_depth = test::count_driver_stack();

  {
    const auto stream = cuda::stream{stream_device};
    auto src          = cuda::make_device_buffer<int>(stream, stream_device, 1, cuda::no_init);
    auto dst          = cuda::make_device_buffer<int>(stream, stream_device, 1, cuda::no_init);

    {
      const auto guard           = cuda::__ensure_current_context{current_device};
      const auto current_context = driver::__ctxGetCurrent();
      const auto stack_depth     = test::count_driver_stack();

      REQUIRE(current_context != driver::__streamGetCtx(stream.get()));

      driver::__memcpyAsync(dst.data(), src.data(), sizeof(*src.data()), stream.get());

      REQUIRE(driver::__ctxGetCurrent() == current_context);
      REQUIRE(test::count_driver_stack() == stack_depth);
    }

    stream.sync();
  }
  REQUIRE(test::count_driver_stack() == initial_stack_depth);
}
