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

// template<class T, class Compare>
//   T
//   max(initializer_list<T> t, Compare comp);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/initializer_list>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int i = cuda::std::max({2, 3, 1}, cuda::std::greater<int>());
  assert(i == 1);
  i = cuda::std::max({2, 1, 3}, cuda::std::greater<int>());
  assert(i == 1);
  i = cuda::std::max({3, 1, 2}, cuda::std::greater<int>());
  assert(i == 1);
  i = cuda::std::max({3, 2, 1}, cuda::std::greater<int>());
  assert(i == 1);
  i = cuda::std::max({1, 2, 3}, cuda::std::greater<int>());
  assert(i == 1);
  i = cuda::std::max({1, 3, 2}, cuda::std::greater<int>());
  assert(i == 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
