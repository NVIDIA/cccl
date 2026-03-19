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

template <class T>
__host__ __device__ constexpr void test()
{
  // Test default
  assert(cuda::__is_valid_alignment(alignof(T)));
  assert(!cuda::__is_valid_alignment(0));
  assert(cuda::__is_valid_alignment(2 * alignof(T)));
  assert(cuda::__is_valid_alignment(4 * alignof(T)));
  assert(!cuda::__is_valid_alignment(alignof(T) + 1) || alignof(T) == 1);
  assert(!cuda::__is_valid_alignment(2 * alignof(T) + 1));

  // Test for type
  assert(cuda::__is_valid_alignment<T>(alignof(T)));
  assert(!cuda::__is_valid_alignment<T>(0));
  assert(cuda::__is_valid_alignment<T>(2 * alignof(T)));
  assert(cuda::__is_valid_alignment<T>(4 * alignof(T)));
  assert(!cuda::__is_valid_alignment<T>(alignof(T) + 1) || alignof(T) == 1);
  assert(!cuda::__is_valid_alignment<T>(2 * alignof(T) + 1));
}

struct alignas(512) OverAligned
{
  char data_[512];
};

__host__ __device__ constexpr bool test()
{
  test<unsigned char>();
  test<short>();
  test<unsigned>();
  test<long long>();
  test<OverAligned>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
