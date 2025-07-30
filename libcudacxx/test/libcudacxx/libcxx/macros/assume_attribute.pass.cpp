//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/attributes.h>

#include "test_macros.h"

__host__ __device__ bool f(int x)
{
  _CCCL_ASSUME(x > 3);
  return x > 3;
}

int main(int, char**)
{
  unused(f(5));
  return 0;
}
