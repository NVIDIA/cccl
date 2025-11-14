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

  auto ctx = _CCCL_TRY_DRIVER_API(__ctxGetCurrent());
  CCCLRT_REQUIRE(ctx != nullptr);

  // Confirm pop will leave the stack empty
  (void) _CCCL_TRY_DRIVER_API(__ctxPop());
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == nullptr);

  // Confirm we can push multiple times
  (void) _CCCL_TRY_DRIVER_API(__ctxPush(ctx));
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == ctx);

  (void) _CCCL_TRY_DRIVER_API(__ctxPush(ctx));
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == ctx);

  _CCCL_ASSERT_DRIVER_API(__ctxPop());
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == ctx);

  // Confirm stream ctx match
  auto stream_ctx = _CCCL_TRY_DRIVER_API(__streamGetCtx(stream));
  CCCLRT_REQUIRE(ctx == stream_ctx);

  CUDART(cudaStreamDestroy(stream));

  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__deviceGet(0)) == 0);

  // Confirm we can retain the primary ctx that cudart retained first
  auto primary_ctx = _CCCL_TRY_DRIVER_API(__primaryCtxRetain(0));
  CCCLRT_REQUIRE(ctx == primary_ctx);

  _CCCL_ASSERT_DRIVER_API(__ctxPop());
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == nullptr);

  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__isPrimaryCtxActive(0)));
  // Confirm we can reset the primary context with double release
  _CCCL_ASSERT_DRIVER_API(__primaryCtxRelease(0));
  _CCCL_ASSERT_DRIVER_API(__primaryCtxRelease(0));

  CCCLRT_REQUIRE(!_CCCL_TRY_DRIVER_API(__isPrimaryCtxActive(0)));

  // Confirm cudart can recover
  CUDART(cudaStreamCreate(&stream));
  CCCLRT_REQUIRE(_CCCL_TRY_DRIVER_API(__ctxGetCurrent()) == ctx);

  _CCCL_ASSERT_DRIVER_API(__streamDestroy(stream));
}
