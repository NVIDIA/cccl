//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.sort_heap.h>
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
  int a[] = {1, 2, 3};
  cuda::std::push_heap(a, a + 1);
  cuda::std::push_heap(a, a + 2);
  cuda::std::push_heap(a, a + 3);
  cuda::std::sort_heap(a, a + 3);
  assert(a[0] == 1 && a[2] == 3);
  int b[] = {1, 2, 3};
  cuda::std::push_heap(b, b + 1);
  cuda::std::push_heap(b, b + 2);
  cuda::std::push_heap(b, b + 3);
  cuda::std::sort_heap(b, b + 3, less{});
  assert(b[0] == 1 && b[2] == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
