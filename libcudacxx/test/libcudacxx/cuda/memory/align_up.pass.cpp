//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <typename T>
TEST_FUNC void test_impl()
{
  if constexpr (alignof(T) <= 2)
  {
    uintptr_t ptr_int = 10;
    auto ptr          = reinterpret_cast<T>(ptr_int);
    assert(cuda::align_up(ptr, 1) == ptr);
    assert(cuda::align_up(ptr, 2) == ptr);
    assert(cuda::align_up(ptr, 4) == reinterpret_cast<T>(12));
    assert(cuda::align_up(ptr, 8) == reinterpret_cast<T>(16));
  }
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<T>(ptr_int2);
  assert(cuda::align_up(ptr2, 8) == reinterpret_cast<T>(16));
}

template <typename T>
TEST_FUNC void test()
{
  test_impl<T*>();
  test_impl<const T*>();
  test_impl<volatile T*>();
  test_impl<const volatile T*>();
}

TEST_FUNC bool test()
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
  assert(cuda::device::is_address_from(cuda::align_up(ptr + 3, 8), cuda::device::address_space::shared));
}

int main(int, char**)
{
  assert(test());
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
