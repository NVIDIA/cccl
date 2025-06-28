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
  assert(cuda::align_down(ptr, 1) == ptr);
  assert(cuda::align_down(ptr, 2) == ptr);
  assert(cuda::align_down(ptr, 4) == ptr - 2);
  assert(cuda::align_down(ptr, 8) == ptr - 2);
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<int*>(ptr_int2);
  assert(cuda::align_down(ptr2, 8) == ptr2 - 1);
  size_t align = 8;
  assert(cuda::align_down(ptr2, align) == ptr2 - 1); // run-time alignment
  return true;
}

__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(__isShared(ptr + 3));
  assert(__isShared(cuda::align_down(ptr + 3, 8)));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
