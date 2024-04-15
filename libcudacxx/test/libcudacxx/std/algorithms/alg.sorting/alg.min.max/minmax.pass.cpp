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
//   pair<const T&, const T&>
//   minmax(const T& a, const T& b);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(const T& a, const T& b, const T& x, const T& y)
{
  cuda::std::pair<const T&, const T&> p = cuda::std::minmax(a, b);
  assert(&p.first == &x);
  assert(&p.second == &y);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  {
    int x = 0;
    int y = 0;
    test(x, y, x, y);
    test(y, x, y, x);
  }
  {
    int x = 0;
    int y = 1;
    test(x, y, x, y);
    test(y, x, x, y);
  }
  {
    int x = 1;
    int y = 0;
    test(x, y, y, x);
    test(y, x, y, x);
  }

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
