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

template <typename T>
__host__ __device__ void test_impl()
{
  if constexpr (alignof(T) <= 2)
  {
    uintptr_t ptr_int = 10;
    auto ptr          = reinterpret_cast<T>(ptr_int);
    assert(cuda::align_down(ptr, 1) == ptr);
    assert(cuda::align_down(ptr, 2) == ptr);
    assert(cuda::align_down(ptr, 4) == reinterpret_cast<T>(8));
    assert(cuda::align_down(ptr, 8) == reinterpret_cast<T>(8));
  }
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<T>(ptr_int2);
  assert(cuda::align_down(ptr2, 8) == reinterpret_cast<T>(8));
  size_t align = 8;
  assert(cuda::align_down(ptr2, align) == reinterpret_cast<T>(8)); // run-time alignment
}

template <typename T>
__host__ __device__ void test()
{
  test_impl<T*>();
  test_impl<const T*>();
  test_impl<volatile T*>();
  test_impl<const volatile T*>();
}

__host__ __device__ bool test()
{
  test<char>();
  test<short>();
  test<int>();
  test<void>();
  return true;
}
__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(cuda::device::is_address_from(ptr + 3, cuda::device::address_space::shared));
  assert(cuda::device::is_address_from(cuda::align_down(ptr + 3, 8), cuda::device::address_space::shared));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
