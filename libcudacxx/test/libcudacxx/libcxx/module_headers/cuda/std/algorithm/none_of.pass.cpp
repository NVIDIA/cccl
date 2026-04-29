//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.none_of.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct none_of_even
{
  __host__ __device__ constexpr bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 3, 5};
  assert(cuda::std::none_of(a, a + 3, none_of_even{}));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
