//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.minmax_element.h>
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
  constexpr int a[] = {3, 1, 2};
  auto p            = cuda::std::minmax_element(a, a + 3);
  assert(*p.first == 1 && *p.second == 3);
  auto q = cuda::std::minmax_element(a, a + 3, greater{});
  assert(*q.first == 3 && *q.second == 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
