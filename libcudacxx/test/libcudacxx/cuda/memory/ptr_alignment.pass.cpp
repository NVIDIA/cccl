//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

template <typename T, typename U>
__host__ __device__ void test()
{
  auto ptr1 = reinterpret_cast<T>(uintptr_t{1});
  assert(cuda::ptr_alignment(ptr1) == 1);

  auto ptr2 = reinterpret_cast<T>(uintptr_t{2});
  assert(cuda::ptr_alignment(ptr2) == 2);

  auto ptr4 = reinterpret_cast<U>(uintptr_t{4});
  assert(cuda::ptr_alignment(ptr4) == 4);

  auto ptr8 = reinterpret_cast<U>(uintptr_t{8});
  assert(cuda::ptr_alignment(ptr8) == 8);

  auto ptr10 = reinterpret_cast<T>(uintptr_t{10});
  assert(cuda::ptr_alignment(ptr10) == 2);

  auto ptr7 = reinterpret_cast<T>(uintptr_t{7});
  assert(cuda::ptr_alignment(ptr7) == 1);

  // max_alignment
  auto ptr12 = reinterpret_cast<U>(uintptr_t{12});
  assert(cuda::ptr_alignment(ptr12) == 4);
  assert(cuda::ptr_alignment(ptr12, 2) == 2);
  assert(cuda::ptr_alignment(ptr12, 4) == 4);
  assert(cuda::ptr_alignment(ptr12, 8) == 4);

  assert(cuda::ptr_alignment(ptr8, 1) == 1);
  assert(cuda::ptr_alignment(ptr1, 1) == 1);
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
