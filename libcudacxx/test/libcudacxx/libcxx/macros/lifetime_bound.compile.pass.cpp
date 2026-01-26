//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/attributes.h>

#include "test_macros.h"

struct S
{
  __host__ __device__ int get() const _CCCL_LIFETIMEBOUND
  {
    return 3;
  }
};

int main(int, char**)
{
  return 0;
}
