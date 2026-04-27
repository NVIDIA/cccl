//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.count_if.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct count_if_even
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3, 4};
  assert(cuda::std::count_if(a, a + 4, count_if_even{}) == 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
