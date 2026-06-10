//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/execution_space.h>

__global__ void test_kernel(const _CCCL_GRID_CONSTANT int value)
{
  (void) value;
}

int main(int, char**)
{
  return 0;
}
