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
  assert(cuda::is_aligned(ptr, 1));
  assert(cuda::is_aligned(ptr, 2));
  assert(!cuda::is_aligned(ptr, 4));
  assert(!cuda::is_aligned(ptr, 8));
  uintptr_t ptr_int2 = 12;
  auto ptr2          = reinterpret_cast<U>(ptr_int2);
  assert(cuda::is_aligned(ptr2, 4));
  assert(!cuda::is_aligned(ptr2, 8));
}

__host__ __device__ bool test()
{
  test<char*, int*>();
  test<const char*, const int*>();
  test<volatile char*, volatile int*>();
  test<const volatile char*, const volatile int*>();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
