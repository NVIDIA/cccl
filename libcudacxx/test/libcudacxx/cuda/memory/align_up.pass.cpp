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

template <typename T, typename U>
__host__ __device__ void test()
{
  uintptr_t ptr_int = 10;
  auto ptr          = reinterpret_cast<T>(ptr_int);
  assert(cuda::align_up(ptr, 1) == ptr);
  assert(cuda::align_up(ptr, 2) == ptr);
  assert(cuda::align_up(ptr, 4) == reinterpret_cast<T>(12));
  assert(cuda::align_up(ptr, 8) == reinterpret_cast<T>(16));
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<U>(ptr_int2);
  assert(cuda::align_up(ptr2, 8) == reinterpret_cast<U>(16));
}

__host__ __device__ bool test()
{
  test<char*, int*>();
  test<const char*, const int*>();
  test<volatile char*, volatile int*>();
  test<const volatile char*, const volatile int*>();
  test<void*, void*>();
  return true;
}

__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(cuda::device::is_address_from(ptr + 3, cuda::device::address_space::shared));
  assert(cuda::device::is_address_from(cuda::align_up(ptr + 3, 8), cuda::device::address_space::shared));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
