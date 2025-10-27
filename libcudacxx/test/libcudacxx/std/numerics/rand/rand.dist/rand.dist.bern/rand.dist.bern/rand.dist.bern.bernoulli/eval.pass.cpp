//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "test_discrete_distribution.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  typedef cuda::std::bernoulli_distribution D;
  typedef cuda::std::minstd_rand G;
  const int n = 10000;
  G g;
  D d;
  auto result = test_discrete_distribution<D, G, 2>(g, d, D::param_type{0.5}, {0.5, 0.5}, n);
  assert(result);
  result = test_discrete_distribution<D, G, 2>(g, d, D::param_type{0.2}, {0.2, 0.8}, n);
  assert(result);
  result = test_discrete_distribution<D, G, 2>(g, d, D::param_type{0.99}, {0.99, 0.01}, n);
  assert(result);
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
