//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// unsigned long to_ulong() const; // constexpr since C++23

#include <cuda/std/bitset>
// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_to_ulong()
{
  const cuda::std::size_t M   = sizeof(unsigned long) * CHAR_BIT < N ? sizeof(unsigned long) * CHAR_BIT : N;
  const bool is_M_zero        = cuda::std::integral_constant<bool, M == 0>::value; // avoid compiler warnings
  const cuda::std::size_t X   = is_M_zero ? sizeof(unsigned long) * CHAR_BIT - 1 : sizeof(unsigned long) * CHAR_BIT - M;
  const cuda::std::size_t max = is_M_zero ? 0 : cuda::std::size_t(cuda::std::numeric_limits<unsigned long>::max()) >> X;
  cuda::std::size_t tests[]   = {
    0,
    cuda::std::min<cuda::std::size_t>(1, max),
    cuda::std::min<cuda::std::size_t>(2, max),
    cuda::std::min<cuda::std::size_t>(3, max),
    cuda::std::min(max, max - 3),
    cuda::std::min(max, max - 2),
    cuda::std::min(max, max - 1),
    max};
  for (cuda::std::size_t j : tests)
  {
    cuda::std::bitset<N> v(j);
    assert(j == v.to_ulong());
  }

  { // test values bigger than can fit into the bitset
    const unsigned long val  = 0x5AFFFFA5UL;
    const bool canFit        = N < sizeof(unsigned long) * CHAR_BIT;
    const unsigned long mask = canFit ? (1UL << (canFit ? N : 0)) - 1 : (unsigned long) (-1); // avoid compiler warnings
    cuda::std::bitset<N> v(val);
    assert(v.to_ulong() == (val & mask)); // we shouldn't return bit patterns from outside the limits of the bitset.
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_to_ulong<0>();
  test_to_ulong<1>();
  test_to_ulong<31>();
  test_to_ulong<32>();
  test_to_ulong<33>();
  test_to_ulong<63>();
  test_to_ulong<64>();
  test_to_ulong<65>();
  test_to_ulong<1000>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && (!defined(_CCCL_CUDACC_BELOW_11_8) || !defined(_CCCL_COMPILER_MSVC))
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
