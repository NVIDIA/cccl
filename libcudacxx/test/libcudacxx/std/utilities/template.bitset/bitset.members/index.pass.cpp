//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>::reference operator[](size_t pos); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_index()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> v1(cases[c]);
    if (v1.size() > 0)
    {
      assert(v1[N / 2] == v1.test(N / 2));
      typename cuda::std::bitset<N>::reference r = v1[N / 2];
      assert(r == v1.test(N / 2));
      typename cuda::std::bitset<N>::reference r2 = v1[N / 2];
      r                                           = r2;
      assert(r == v1.test(N / 2));
      r = false;
      assert(r == false);
      assert(v1.test(N / 2) == false);
      r = true;
      assert(r == true);
      assert(v1.test(N / 2) == true);
      bool b = ~r;
      assert(r == true);
      assert(v1.test(N / 2) == true);
      assert(b == false);
      r.flip();
      assert(r == false);
      assert(v1.test(N / 2) == false);
    }
    ASSERT_SAME_TYPE(decltype(v1[0]), typename cuda::std::bitset<N>::reference);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_index<0>();
  test_index<1>();
  test_index<31>();
  test_index<32>();
  test_index<33>();
  test_index<63>();
  test_index<64>();
  test_index<65>();

  cuda::std::bitset<1> set;
  set[0] = false;
  auto b = set[0];
  set[0] = true;
  assert(b);

  return true;
}

int main(int, char**)
{
  test();
  test_index<1000>(); // not in constexpr because of constexpr evaluation step limits
// 11.4 added support for constexpr device vars needed here
#if TEST_STD_VER >= 2014 && !defined(_CCCL_CUDACC_BELOW_11_4)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
