//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// size_t count() const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_size()
{
  const cuda::std::bitset<N> v;
  assert(v.size() == N);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_size<0>();
  test_size<1>();
  test_size<31>();
  test_size<32>();
  test_size<33>();
  test_size<63>();
  test_size<64>();
  test_size<65>();
  test_size<1000>();

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
