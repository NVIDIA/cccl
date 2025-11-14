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
//   pair<T, T>
//   minmax(initializer_list<T> t);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  assert((cuda::std::minmax({1, 2, 3}) == cuda::std::pair<int, int>(1, 3)));
  assert((cuda::std::minmax({1, 3, 2}) == cuda::std::pair<int, int>(1, 3)));
  assert((cuda::std::minmax({2, 1, 3}) == cuda::std::pair<int, int>(1, 3)));
  assert((cuda::std::minmax({2, 3, 1}) == cuda::std::pair<int, int>(1, 3)));
  assert((cuda::std::minmax({3, 1, 2}) == cuda::std::pair<int, int>(1, 3)));
  assert((cuda::std::minmax({3, 2, 1}) == cuda::std::pair<int, int>(1, 3)));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
