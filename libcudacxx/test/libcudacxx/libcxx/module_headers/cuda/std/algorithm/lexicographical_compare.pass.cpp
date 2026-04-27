//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.lexicographical_compare.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct less
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a < b;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2};
  constexpr int b[] = {1, 3};
  assert(cuda::std::lexicographical_compare(a, a + 2, b, b + 2));
  assert(cuda::std::lexicographical_compare(a, a + 2, b, b + 2, less{}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
