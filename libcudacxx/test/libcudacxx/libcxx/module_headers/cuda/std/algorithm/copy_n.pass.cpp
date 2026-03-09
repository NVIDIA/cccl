//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.copy_n.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  constexpr int a[] = {5, 6, 7};
  int o[2]          = {};
  auto r            = cuda::std::copy_n(a, 2, o);
  assert(r == o + 2 && o[0] == 5 && o[1] == 6);

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
