//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.copy.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3};
  int o[3]          = {};
  auto r            = cuda::std::copy(a, a + 3, o);
  assert(r == o + 3 && o[0] == 1 && o[2] == 3);

  return true;
}

int main(int, char**)
{
  test();

#if !TEST_COMPILER(GCC, <, 8)
  static_assert(test());
#endif // !TEST_COMPILER(GCC, <, 8)

  return 0;
}
