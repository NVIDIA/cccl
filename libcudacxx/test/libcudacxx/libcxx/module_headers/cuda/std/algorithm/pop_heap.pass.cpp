//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.pop_heap.h>
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
  int a[] = {2, 1, 3};
  cuda::std::pop_heap(a, a + 3);
  assert(a[0] == 3 && a[1] == 1 && a[2] == 2);

  int b[] = {2, 1, 3};
  cuda::std::pop_heap(b, b + 3, less{});
  assert(b[0] == 3 && b[1] == 1 && b[2] == 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
