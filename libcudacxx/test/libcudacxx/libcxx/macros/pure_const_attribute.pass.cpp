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

_CCCL_PURE __host__ __device__ int f()
{
  return 0;
}

_CCCL_CONST __host__ __device__ int g()
{
  return 0;
}

int main(int, char**)
{
  unused(f());
  return 0;
}
