//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.remove_copy_if.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct remove_copy_if_is_odd
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 1;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3};
  int o[3]          = {};
  auto r            = cuda::std::remove_copy_if(a, a + 3, o, remove_copy_if_is_odd{});
  assert(r == o + 1 && o[0] == 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
