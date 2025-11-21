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
  cudaStream_t stream;
  // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
  // test leaves 2+ ctxs on the stack

  // Pushes the primary context if the stack is empty
  CUDART(cudaStreamCreate(&stream));

  auto ctx = CCCLRT_DRIVER_CALL(__ctxGetCurrent());
  CCCLRT_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  (void) CCCLRT_DRIVER_CALL(__ctxPop());
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == nullptr);

  // Confirm we can push multiple times
  (void) CCCLRT_DRIVER_CALL(__ctxPush(ctx));
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == ctx);

  (void) CCCLRT_DRIVER_CALL(__ctxPush(ctx));
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == ctx);

  (void) CCCLRT_DRIVER_CALL(__ctxPop());
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == ctx);

  // Confirm stream ctx match
  auto stream_ctx = CCCLRT_DRIVER_CALL(__streamGetCtx(stream));
  CCCLRT_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__deviceGet(0)) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = CCCLRT_DRIVER_CALL(__primaryCtxRetain(0));
  CCCLRT_REQUIRE(ctx == primary_ctx);

  (void) CCCLRT_DRIVER_CALL(__ctxPop());
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == nullptr);

  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__isPrimaryCtxActive(0)));
  // Confirm we can reset the primary context with double release
  CCCLRT_DRIVER_CALL(__primaryCtxRelease(0));
  CCCLRT_DRIVER_CALL(__primaryCtxRelease(0));

  CCCLRT_REQUIRE(!CCCLRT_DRIVER_CALL(__isPrimaryCtxActive(0)));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(CCCLRT_DRIVER_CALL(__ctxGetCurrent()) == ctx);

  CCCLRT_DRIVER_CALL(__streamDestroy(stream));
}
