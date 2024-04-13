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
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ BITSET_TEST_CONSTEXPR void test_count()
{
  span_stub<const char*> const cases = get_test_cases<N>();
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    const cuda::std::bitset<N> v(cases[c]);
    cuda::std::size_t c1 = v.count();
    cuda::std::size_t c2 = 0;
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      if (v[i])
      {
        ++c2;
      }
    }
    assert(c1 == c2);
  }
}

__host__ __device__ BITSET_TEST_CONSTEXPR bool test()
{
  test_count<0>();
  test_count<1>();
  test_count<31>();
  test_count<32>();
  test_count<33>();
  test_count<63>();
  test_count<64>();
  test_count<65>();

  return true;
}

int main(int, char**)
{
  test();
  test_count<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 2011 && !defined(_LIBCUDACXX_CUDACC_BELOW_11_4) // 11.4 added support for constexpr device vars
                                                                   // needed here
  static_assert(test(), "");
#endif

  return 0;
}
