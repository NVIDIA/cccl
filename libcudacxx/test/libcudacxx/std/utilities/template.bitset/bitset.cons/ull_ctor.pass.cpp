//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset(unsigned long long val); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
// #include <cuda/std/algorithm> // for 'min' and 'max'
#include <cuda/std/cstddef>

#include "test_macros.h"

// TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not
// executed.
TEST_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_val_ctor()
{
  {
    constexpr cuda::std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
    assert(v.size() == N);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 64);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == ((i & 1) != 0));
    }
    for (cuda::std::size_t i = M; i < v.size(); ++i)
    {
      {
        assert(v[i] == false);
      }
    }
  }
  {
    constexpr cuda::std::bitset<N> v(0xAAAAAAAAAAAAAAAAULL);
    static_assert(v.size() == N, "");
  }
}

__host__ __device__ constexpr bool test()
{
  test_val_ctor<0>();
  test_val_ctor<1>();
  test_val_ctor<31>();
  test_val_ctor<32>();
  test_val_ctor<33>();
  test_val_ctor<63>();
  test_val_ctor<64>();
  test_val_ctor<65>();
  test_val_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
