//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// unsigned long long to_ullong() const; // constexpr since C++23

#include <cuda/std/bitset>
// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_to_ullong()
{
  const cuda::std::size_t M = sizeof(unsigned long long) * CHAR_BIT < N ? sizeof(unsigned long long) * CHAR_BIT : N;
  const bool is_M_zero      = cuda::std::integral_constant<bool, M == 0>::value; // avoid compiler warnings
  const cuda::std::size_t X =
    is_M_zero ? sizeof(unsigned long long) * CHAR_BIT - 1 : sizeof(unsigned long long) * CHAR_BIT - M;
  const unsigned long long max = is_M_zero ? 0 : (unsigned long long) (-1) >> X;
  unsigned long long tests[]   = {
    0,
    cuda::std::min<unsigned long long>(1, max),
    cuda::std::min<unsigned long long>(2, max),
    cuda::std::min<unsigned long long>(3, max),
    cuda::std::min(max, max - 3),
    cuda::std::min(max, max - 2),
    cuda::std::min(max, max - 1),
    max};
  for (unsigned long long j : tests)
  {
    cuda::std::bitset<N> v(j);
    assert(j == v.to_ullong());
  }
  { // test values bigger than can fit into the bitset
    const unsigned long long val  = 0x55AAAAFFFFAAAA55ULL;
    const bool canFit             = N < sizeof(unsigned long long) * CHAR_BIT;
    const unsigned long long mask = canFit ? (1ULL << (canFit ? N : 0)) - 1 : (unsigned long long) (-1); // avoid
                                                                                                         // compiler
                                                                                                         // warnings
    cuda::std::bitset<N> v(val);
    assert(v.to_ullong() == (val & mask)); // we shouldn't return bit patterns from outside the limits of the bitset.
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_to_ullong<0>();
  test_to_ullong<1>();
  test_to_ullong<31>();
  test_to_ullong<32>();
  test_to_ullong<33>();
  test_to_ullong<63>();
  test_to_ullong<64>();
  test_to_ullong<65>();
  test_to_ullong<1000>();

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
