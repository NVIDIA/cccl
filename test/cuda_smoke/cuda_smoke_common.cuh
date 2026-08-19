//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#define CUDART_REQUIRE(call) REQUIRE((call) == cudaSuccess)

__global__ void increment_kernel(int* p, int n)
{
  int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n)
  {
    p[idx] += 1;
  }
}
