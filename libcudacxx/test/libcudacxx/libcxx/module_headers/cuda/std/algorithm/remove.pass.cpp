//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.remove.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int a[] = {1, 2, 1, 3};
  auto r  = cuda::std::remove(a, a + 4, 1);
  assert(r == a + 2 && a[0] == 2 && a[1] == 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
