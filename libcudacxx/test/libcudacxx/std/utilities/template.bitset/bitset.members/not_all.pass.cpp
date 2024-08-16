//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N> operator~() const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_not_all()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> v1(cases[c]);
    cuda::std::bitset<N> v2 = ~v1;
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_ICC(186)
    for (cuda::std::size_t i = 0; i < v1.size(); ++i)
    {
      _CCCL_DIAG_POP
      {
        assert(v2[i] == ~v1[i]);
      }
    }
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_not_all<0>();
  test_not_all<1>();
  test_not_all<31>();
  test_not_all<32>();
  test_not_all<33>();
  test_not_all<63>();
  test_not_all<64>();
  test_not_all<65>();

  return true;
}

int main(int, char**)
{
  test();
  test_not_all<1000>(); // not in constexpr because of constexpr evaluation step limits
// 11.4 added support for constexpr device vars needed here
#if TEST_STD_VER >= 2014 && !defined(_CCCL_CUDACC_BELOW_11_4)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
