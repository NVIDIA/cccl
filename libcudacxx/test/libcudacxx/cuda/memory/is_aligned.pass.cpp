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

__host__ __device__ bool test()
{
  uintptr_t ptr_int = 10;
  auto ptr          = reinterpret_cast<char*>(ptr_int);
  assert(cuda::is_aligned(ptr, 1));
  assert(cuda::is_aligned(ptr, 2));
  assert(!cuda::is_aligned(ptr, 4));
  assert(!cuda::is_aligned(ptr, 8));
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<int*>(ptr_int2);
  assert(cuda::is_aligned(ptr2, 4));
  assert(!cuda::is_aligned(ptr2, 8));
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
