//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool none() const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_none()
{
  cuda::std::bitset<N> v;
  v.reset();
  assert(v.none() == true);
  v.set();
  assert(v.none() == (N == 0));
  if (v.size() > 1)
  {
    v[N / 2] = false;
    assert(v.none() == false);
    v.reset();
    v[N / 2] = true;
    assert(v.none() == false);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_none<0>();
  test_none<1>();
  test_none<31>();
  test_none<32>();
  test_none<33>();
  test_none<63>();
  test_none<64>();
  test_none<65>();
  test_none<1000>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 7)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && !gcc-6

  return 0;
}
