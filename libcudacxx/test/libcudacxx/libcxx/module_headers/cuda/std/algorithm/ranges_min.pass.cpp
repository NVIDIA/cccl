//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.min.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  const int one{1};
  const int two{2};
  const int three{3};
  const int four{4};
  const int five{5};
  const int eight{8};

  assert(cuda::std::ranges::min(one, two) == one);
  assert(cuda::std::ranges::min(three, one, cuda::std::ranges::greater{}) == three);
  int a[] = {three, one, four};
  assert(cuda::std::ranges::min(a) == one);
  assert(cuda::std::ranges::min({five, two, eight}) == two);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
