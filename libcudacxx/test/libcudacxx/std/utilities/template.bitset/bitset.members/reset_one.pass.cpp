//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& reset(size_t pos); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

// TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not
// executed.

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_reset_one()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  if (Start >= 9)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c = Start; c != cases.size() && c != End; ++c)
  {
    for (cuda::std::size_t i = 0; i != N; ++i)
    {
      cuda::std::bitset<N> v(cases[c]);
      v.reset(i);
      assert(v[i] == false);
    }
  }

  return true;
}

int main(int, char**)
{
  test_reset_one<0>();
  test_reset_one<1>();
  test_reset_one<31>();
  test_reset_one<32>();
  test_reset_one<33>();
  test_reset_one<63>();
  test_reset_one<64>();
  test_reset_one<65>();
  test_reset_one<1000>(); // not in constexpr because of constexpr evaluation step limits
// 11.4 added support for constexpr device vars needed here
#if TEST_STD_VER >= 2014 && !defined(_CCCL_CUDACC_BELOW_11_4)
  static_assert(test_reset_one<0>(), "");
  static_assert(test_reset_one<1>(), "");
  static_assert(test_reset_one<31>(), "");
  static_assert(test_reset_one<32>(), "");
  static_assert(test_reset_one<33>(), "");
  static_assert(test_reset_one<63, 0, 6>(), "");
  static_assert(test_reset_one<63, 6>(), "");
  static_assert(test_reset_one<64, 0, 3>(), "");
  static_assert(test_reset_one<64, 3, 6>(), "");
  static_assert(test_reset_one<64, 6, 9>(), "");
  static_assert(test_reset_one<64, 9>(), "");
  static_assert(test_reset_one<65, 0, 3>(), "");
  static_assert(test_reset_one<65, 3, 6>(), "");
  static_assert(test_reset_one<65, 6, 9>(), "");
  static_assert(test_reset_one<65, 9>(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
