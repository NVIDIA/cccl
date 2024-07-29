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

// template<class T, StrictWeakOrder<auto, T> Compare>
//   requires !SameType<T, Compare> && CopyConstructible<Compare>
//   const T&
//   max(const T& a, const T& b, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_macros.h"

template <class T, class C>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(const T& a, const T& b, C c, const T& x)
{
  assert(&cuda::std::max(a, b, c) == &x);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  {
    int x = 0;
    int y = 0;
    test(x, y, cuda::std::greater<int>(), x);
    test(y, x, cuda::std::greater<int>(), y);
  }
  {
    int x = 0;
    int y = 1;
    test(x, y, cuda::std::greater<int>(), x);
    test(y, x, cuda::std::greater<int>(), x);
  }
  {
    int x = 1;
    int y = 0;
    test(x, y, cuda::std::greater<int>(), y);
    test(y, x, cuda::std::greater<int>(), y);
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
  static_assert(&cuda::std::max(x, y, cuda::std::greater<int>()) == &x, "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
