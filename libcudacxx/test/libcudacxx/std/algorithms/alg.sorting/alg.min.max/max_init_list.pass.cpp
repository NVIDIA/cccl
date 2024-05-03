//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template <class T>
//   T
//   max(initializer_list<T> t);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int i = cuda::std::max({2, 3, 1});
  assert(i == 3);
  i = cuda::std::max({2, 1, 3});
  assert(i == 3);
  i = cuda::std::max({3, 1, 2});
  assert(i == 3);
  i = cuda::std::max({3, 2, 1});
  assert(i == 3);
  i = cuda::std::max({1, 2, 3});
  assert(i == 3);
  i = cuda::std::max({1, 3, 2});
  assert(i == 3);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
