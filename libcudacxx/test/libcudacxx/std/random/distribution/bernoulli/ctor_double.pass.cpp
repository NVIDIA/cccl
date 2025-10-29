//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// explicit bernoulli_distribution(double p = 0.5);          // before C++20
// bernoulli_distribution() : bernoulli_distribution(0.5) {} // C++20
// explicit bernoulli_distribution(double p);                // C++20

#include <cuda/std/__random_>

#include <cassert>

#include "make_implicit.h"
#include "test_convertible.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    typedef cuda::std::bernoulli_distribution D;
    D d;
    assert(d.p() == 0.5);
  }
  {
    typedef cuda::std::bernoulli_distribution D;
    D d(0);
    assert(d.p() == 0);
  }
  {
    typedef cuda::std::bernoulli_distribution D;
    D d(0.75);
    assert(d.p() == 0.75);
  }

  {
    typedef cuda::std::bernoulli_distribution D;
    static_assert(test_convertible<D>());
    assert(D(0.5) == make_implicit<D>());
    static_assert(!test_convertible<D, double>());
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif

  return 0;
}
