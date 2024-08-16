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

// template<LessThanComparable T>
//   const T&
//   min(const T& a, const T& b);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(const T& a, const T& b, const T& x)
{
  assert(&cuda::std::min(a, b) == &x);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  {
    int x = 0;
    int y = 0;
    test(x, y, x);
    test(y, x, y);
  }
  {
    int x = 0;
    int y = 1;
    test(x, y, x);
    test(y, x, x);
  }
  {
    int x = 1;
    int y = 0;
    test(x, y, y);
    test(y, x, y);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#else // TEST_STD_VER >= 2014
  constexpr int x = 0;
  constexpr int y = 1;
  static_assert(&cuda::std::min(x, y) == &x, "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
