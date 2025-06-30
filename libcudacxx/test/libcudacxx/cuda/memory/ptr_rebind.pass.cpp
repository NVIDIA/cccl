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

__host__ __device__ bool test()
{
  uintptr_t ptr_int = 16;
  auto ptr          = reinterpret_cast<char*>(ptr_int);
  assert(cuda::ptr_rebind<uint16_t>(ptr) == (uint16_t*) ptr);
  assert(cuda::ptr_rebind<int>(ptr) == (int*) ptr);
  assert(cuda::ptr_rebind<uint64_t>(ptr) == (uint64_t*) ptr);
  static_assert(cuda::std::is_same_v<int*, decltype(cuda::ptr_rebind<int>(ptr))>);

  auto const_ptr = reinterpret_cast<const char*>(ptr_int);
  assert(cuda::ptr_rebind<uint16_t>(const_ptr) == (const uint16_t*) ptr);
  static_assert(cuda::std::is_same_v<const int*, decltype(cuda::ptr_rebind<int>(const_ptr))>);

  auto ptr2 = reinterpret_cast<void*>(ptr_int);
  assert(cuda::ptr_rebind<uint16_t>(ptr2) == (uint16_t*) ptr);
  return true;
}

__global__ void test_kernel()
{
  __shared__ int smem_value[4];
  auto ptr = smem_value;
  assert(__isShared(cuda::ptr_rebind<uint64_t>(ptr)));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
