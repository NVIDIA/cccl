//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__utility/driver_api.cuh>

#include <testing.cuh>

TEST_CASE("Call each driver api", "[utility]")
{
  namespace driver = cuda::experimental::detail::driver;
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  CUDART(cudaStreamCreate(&stream));

  auto ctx = driver::ctxGetCurrent();
  CUDAX_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  driver::ctxPop();
  CUDAX_REQUIRE(driver::ctxGetCurrent() == nullptr);

  // Confirm we can push multiple times
  driver::ctxPush(ctx);
  CUDAX_REQUIRE(driver::ctxGetCurrent() == ctx);

  driver::ctxPush(ctx);
  CUDAX_REQUIRE(driver::ctxGetCurrent() == ctx);

  driver::ctxPop();
  CUDAX_REQUIRE(driver::ctxGetCurrent() == ctx);

  // Confirm stream ctx match
  auto stream_ctx = driver::streamGetCtx(stream);
  CUDAX_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CUDAX_REQUIRE(driver::deviceGet(0) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = driver::primaryCtxRetain(0);
  CUDAX_REQUIRE(ctx == primary_ctx);

  driver::ctxPop();
  CUDAX_REQUIRE(driver::ctxGetCurrent() == nullptr);

  CUDAX_REQUIRE(driver::isPrimaryCtxActive(0));
  // Confirm we can reset the primary context with double release
  driver::primaryCtxRelease(0);
  driver::primaryCtxRelease(0);

  CUDAX_REQUIRE(!driver::isPrimaryCtxActive(0));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CUDAX_REQUIRE(driver::ctxGetCurrent() == ctx);

  CUDART(driver::streamDestroy(stream));
}
