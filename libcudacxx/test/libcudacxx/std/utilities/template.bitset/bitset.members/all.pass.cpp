//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool all() const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_all()
{
  cuda::std::bitset<N> v;
  v.reset();
  assert(v.all() == (N == 0));
  v.set();
  assert(v.all() == true);
  if (v.size() > 1)
  {
    v[N / 2] = false;
    assert(v.all() == false);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_all<0>();
  test_all<1>();
  test_all<31>();
  test_all<32>();
  test_all<33>();
  test_all<63>();
  test_all<64>();
  test_all<65>();
  test_all<1000>();

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
