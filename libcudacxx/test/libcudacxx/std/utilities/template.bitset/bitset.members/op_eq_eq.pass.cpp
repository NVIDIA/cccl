//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// bool operator==(const bitset<N>& rhs) const; // constexpr since C++23
// bool operator!=(const bitset<N>& rhs) const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_equality()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> const v1(cases[c]);
    cuda::std::bitset<N> v2 = v1;
    assert(v1 == v2);
    if (v1.size() > 0)
    {
      v2[N / 2].flip();
      assert(v1 != v2);
    }
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_equality<0>();
  test_equality<1>();
  test_equality<31>();
  test_equality<32>();
  test_equality<33>();
  test_equality<63>();
  test_equality<64>();
  test_equality<65>();

  return true;
}

int main(int, char**)
{
  test();
  test_equality<1000>(); // not in constexpr because of constexpr evaluation step limits
// 11.4 added support for constexpr device vars needed here
#if TEST_STD_VER >= 2014 && !defined(_CCCL_CUDACC_BELOW_11_4)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
