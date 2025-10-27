//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// bool operator=(const bernoulli_distribution& x,
//                const bernoulli_distribution& y);
// bool operator!(const bernoulli_distribution& x,
//                const bernoulli_distribution& y);

#include <cuda/std/__random_>

#include <cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    typedef cuda::std::bernoulli_distribution D;
    D d1(.25);
    D d2(.25);
    assert(d1 == d2);
  }
  {
    typedef cuda::std::bernoulli_distribution D;
    D d1(.28);
    D d2(.25);
    assert(d1 != d2);
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
