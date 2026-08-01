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

#include <testing.cuh>

// This test is an exception and shouldn't use C2H_CCCLRT_TEST macro
C2H_TEST("Call each driver api", "[utility]")
{
  namespace driver = ::cuda::__driver;
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  REQUIRE_CUDART(cudaStreamCreate(&stream));

  auto ctx = driver::__ctxGetCurrent();
  REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  driver::__ctxPop();
  REQUIRE(driver::__ctxGetCurrent() == nullptr);

  // Confirm we can push multiple times
  driver::__ctxPush(ctx);
  REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPush(ctx);
  REQUIRE(driver::__ctxGetCurrent() == ctx);

  driver::__ctxPop();
  REQUIRE(driver::__ctxGetCurrent() == ctx);

  // Confirm stream ctx match
  auto stream_ctx = driver::__streamGetCtx(stream);
  REQUIRE(ctx == stream_ctx);

  REQUIRE_CUDART(cudaStreamDestroy(stream));

  REQUIRE(driver::__deviceGet(0) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = driver::__primaryCtxRetain(0);
  REQUIRE(ctx == primary_ctx);

  driver::__ctxPop();
  REQUIRE(driver::__ctxGetCurrent() == nullptr);

  REQUIRE(driver::__isPrimaryCtxActive(0));
  // Confirm we can reset the primary context with double release
  REQUIRE_CUDART(driver::__primaryCtxReleaseNoThrow(0));
  REQUIRE_CUDART(driver::__primaryCtxReleaseNoThrow(0));

  // Try a third release in case curand retained the primary ctx as well
  if (driver::__isPrimaryCtxActive(0))
  {
    REQUIRE_CUDART(driver::__primaryCtxReleaseNoThrow(0));
  }

  REQUIRE(!driver::__isPrimaryCtxActive(0));

  // Confirm cudart can recover
  REQUIRE_CUDART(cudaStreamCreate(&stream));
  REQUIRE(driver::__ctxGetCurrent() == ctx);

  REQUIRE_CUDART(driver::__streamDestroyNoThrow(stream));
}
