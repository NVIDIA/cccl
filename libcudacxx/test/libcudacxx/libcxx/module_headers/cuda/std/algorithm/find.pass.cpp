//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.find.h>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  constexpr int arr[] = {2, 4, 6, 8};

  const int* iter = cuda::std::find(arr, arr + 4, 2);
  assert(*iter == 2);
  assert(iter == arr);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
