//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.partial_sort_copy.h>
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
  constexpr int src[] = {3, 1, 4};
  int dst[2]          = {};
  cuda::std::partial_sort_copy(src, src + 3, dst, dst + 2);
  assert(dst[0] == 1 && dst[1] == 3);
  int dst2[2] = {};
  cuda::std::partial_sort_copy(src, src + 3, dst2, dst2 + 2, greater{});
  assert(dst2[0] == 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
