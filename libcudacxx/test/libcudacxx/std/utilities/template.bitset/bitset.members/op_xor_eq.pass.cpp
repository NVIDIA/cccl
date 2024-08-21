//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& operator^=(const bitset<N>& rhs); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_op_xor_eq()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  if (Start >= 9)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c1 = Start; c1 != cases.size() && c1 != End; ++c1)
  {
    for (cuda::std::size_t c2 = 0; c2 != cases.size(); ++c2)
    {
      cuda::std::bitset<N> v1(cases[c1]);
      cuda::std::bitset<N> v2(cases[c2]);
      cuda::std::bitset<N> v3 = v1;
      v1 ^= v2;
      _CCCL_DIAG_PUSH
      _CCCL_DIAG_SUPPRESS_ICC(186)
      for (cuda::std::size_t i = 0; i < v1.size(); ++i)
      {
        _CCCL_DIAG_POP
        {
          assert(v1[i] == (v3[i] != v2[i]));
        }
      }
    }
  }
  return true;
}

int main(int, char**)
{
  test_op_xor_eq<0>();
  test_op_xor_eq<1>();
  test_op_xor_eq<31>();
  test_op_xor_eq<32>();
  test_op_xor_eq<33>();
  test_op_xor_eq<63>();
  test_op_xor_eq<64>();
  test_op_xor_eq<65>();
  test_op_xor_eq<1000>(); // not in constexpr because of constexpr evaluation step limits
// 11.4 added support for constexpr device vars needed here
#if TEST_STD_VER >= 2014 && !defined(_CCCL_CUDACC_BELOW_11_4)
  static_assert(test_op_xor_eq<0>(), "");
  static_assert(test_op_xor_eq<1>(), "");
  static_assert(test_op_xor_eq<31>(), "");
  static_assert(test_op_xor_eq<32>(), "");
  static_assert(test_op_xor_eq<33>(), "");
  static_assert(test_op_xor_eq<63, 0, 6>(), "");
  static_assert(test_op_xor_eq<63, 6>(), "");
  static_assert(test_op_xor_eq<64, 0, 6>(), "");
  static_assert(test_op_xor_eq<64, 6>(), "");
  static_assert(test_op_xor_eq<65, 0, 6>(), "");
  static_assert(test_op_xor_eq<65, 6>(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
