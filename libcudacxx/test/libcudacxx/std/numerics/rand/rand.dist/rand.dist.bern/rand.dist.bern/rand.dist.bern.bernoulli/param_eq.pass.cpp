//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution
// {
//     class param_type;

#include <cuda/std/__random_>

#include <cassert>
#include <limits>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    typedef cuda::std::bernoulli_distribution D;
    typedef D::param_type param_type;
    param_type p1(0.75);
    param_type p2(0.75);
    assert(p1 == p2);
  }
  {
    typedef cuda::std::bernoulli_distribution D;
    typedef D::param_type param_type;
    param_type p1(0.75);
    param_type p2(0.5);
    assert(p1 != p2);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif

  return 0;
}
