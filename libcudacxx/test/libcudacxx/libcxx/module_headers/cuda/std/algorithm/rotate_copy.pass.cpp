//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.rotate_copy.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {1, 2, 3, 4};
  int o[4]          = {};
  auto r            = cuda::std::rotate_copy(a, a + 2, a + 4, o);
  assert(r == o + 4 && o[0] == 3 && o[3] == 2);

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
