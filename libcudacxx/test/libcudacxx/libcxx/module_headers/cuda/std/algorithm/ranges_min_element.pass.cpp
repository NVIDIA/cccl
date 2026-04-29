//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.ranges.min_element.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int a[] = {3, 1, 4};
  assert(*cuda::std::ranges::min_element(a, a + 3) == 1);
  assert(*cuda::std::ranges::min_element(a, a + 3, cuda::std::ranges::greater{}) == 4);
  assert(*cuda::std::ranges::min_element(a) == 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
