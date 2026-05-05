//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.replace_if.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct replace_if_is_odd
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 1;
  }
};

__host__ __device__ constexpr bool test()
{
  int a[] = {1, 2, 3};
  cuda::std::replace_if(a, a + 3, replace_if_is_odd{}, 0);
  assert(a[0] == 0 && a[1] == 2 && a[2] == 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
