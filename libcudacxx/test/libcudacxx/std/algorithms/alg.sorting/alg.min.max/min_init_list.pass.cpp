//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class T>
//   T
//   min(initializer_list<T> t);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int i = cuda::std::min({2, 3, 1});
  assert(i == 1);
  i = cuda::std::min({2, 1, 3});
  assert(i == 1);
  i = cuda::std::min({3, 1, 2});
  assert(i == 1);
  i = cuda::std::min({3, 2, 1});
  assert(i == 1);
  i = cuda::std::min({1, 2, 3});
  assert(i == 1);
  i = cuda::std::min({1, 3, 2});
  assert(i == 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
