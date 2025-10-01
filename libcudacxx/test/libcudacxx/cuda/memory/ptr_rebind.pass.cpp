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
#include <cuda/std/type_traits>

template <typename T, typename U, typename V>
__host__ __device__ void test()
{
  uintptr_t ptr_int         = 16;
  [[maybe_unused]] auto ptr = reinterpret_cast<T>(ptr_int);
  static_assert(cuda::std::is_same_v<U, decltype(cuda::ptr_rebind<V>(ptr))>);
}

__host__ __device__ bool test()
{
  test<char*, char*, char>();
  test<char*, short*, short>();
  test<char*, int*, int>();
  test<char*, void*, void>();
  test<const char*, const int*, int>();
  test<volatile char*, volatile int*, int>();
  test<const volatile char*, const volatile int*, int>();
  test<const char*, const void*, void>();
  return true;
}

__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(cuda::device::is_address_from(cuda::ptr_rebind<uint64_t>(ptr), cuda::device::address_space::shared));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
