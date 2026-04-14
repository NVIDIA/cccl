//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

C2H_TEST("basic stf logical_data", "[logical_data]")
{
  size_t N = 1000000;

  stf_ctx_handle ctx = stf_ctx_create();
  REQUIRE(ctx != nullptr);

  std::vector<float> A(N);
  std::vector<float> B(N);

  stf_logical_data_handle lA = stf_logical_data(ctx, A.data(), N * sizeof(float));
  stf_logical_data_handle lB = stf_logical_data(ctx, B.data(), N * sizeof(float));
  REQUIRE(lA != nullptr);
  REQUIRE(lB != nullptr);

  stf_logical_data_destroy(lA);
  stf_logical_data_destroy(lB);

  stf_ctx_finalize(ctx);
}
