//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.copy_if.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct copy_if_even
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3, 4};
  int o[4]          = {};
  auto r            = cuda::std::copy_if(a, a + 4, o, copy_if_even{});
  assert(r == o + 2 && o[0] == 2 && o[1] == 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
