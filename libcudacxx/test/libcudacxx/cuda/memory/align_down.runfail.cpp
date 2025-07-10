//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

__host__ __device__ bool test()
{
  uintptr_t ptr_int = 10;
  auto ptr          = reinterpret_cast<int*>(ptr_int);
  unused(cuda::align_down(ptr, 7)); // not power of two
  unused(cuda::align_down(ptr, 2)); // alignment smaller than alignof(int)
  unused(cuda::align_down(ptr, 4)); // wrong pointer alignment
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
