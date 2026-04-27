//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.equal.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct equal_to
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a == b;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3};
  constexpr int b[] = {1, 2, 3};
  constexpr int c[] = {1, 2, 4};
  assert(cuda::std::equal(a, a + 3, b));
  assert(cuda::std::equal(a, a + 3, b, equal_to{}));
  assert(cuda::std::equal(a, a + 3, b, b + 3));
  assert(!cuda::std::equal(a, a + 3, c, c + 3, equal_to{}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
