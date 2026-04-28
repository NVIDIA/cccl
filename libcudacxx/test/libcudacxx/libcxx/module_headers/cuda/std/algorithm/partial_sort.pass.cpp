//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.partial_sort.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct greater
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a > b;
  }
};

__host__ __device__ constexpr bool test()
{
  int a[] = {3, 1, 4, 2};
  cuda::std::partial_sort(a, a + 2, a + 4);
  assert(a[0] == 1 && a[1] == 2);
  int b[] = {3, 1, 4, 2};
  cuda::std::partial_sort(b, b + 2, b + 4, greater{});
  assert(b[0] == 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
