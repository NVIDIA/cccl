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
  driver::__primaryCtxRelease(0);
  driver::__primaryCtxRelease(0);

  CCCLRT_REQUIRE(!driver::__isPrimaryCtxActive(0));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ctx);

  CUDART(driver::__streamDestroyNoThrow(stream));
}
