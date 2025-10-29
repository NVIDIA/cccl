//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// explicit bernoulli_distribution(const param_type& parm);

#include <cuda/std/__random_>

#include <cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    typedef cuda::std::bernoulli_distribution D;
    typedef D::param_type P;
    P p(0.25);
    D d(p);
    assert(d.p() == 0.25);
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
