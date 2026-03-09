//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.rotate.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int a[] = {1, 2, 3, 4};
  auto r  = cuda::std::rotate(a, a + 2, a + 4);
  assert(r == a + 2 && a[0] == 3 && a[1] == 4 && a[2] == 1 && a[3] == 2);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
