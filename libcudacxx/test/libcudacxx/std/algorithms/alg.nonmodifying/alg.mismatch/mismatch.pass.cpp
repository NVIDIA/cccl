//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2);
//
// template<InputIterator Iter1, InputIterator Iter2Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2); // C++14

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

#ifdef TEST_COMPILER_MSVC
#  pragma warning(disable : 4018) // signed/unsigned mismatch
#endif // TEST_COMPILER_MSVC

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[]          = {0, 1, 2, 3, 0, 1, 2, 3};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  ((void) sb); // unused in C++11

  typedef cpp17_input_iterator<const int*> II;
  typedef random_access_iterator<const int*> RAI;

  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib)) == (cuda::std::pair<II, II>(II(ia + 3), II(ib + 3))));

  assert(cuda::std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib)) == (cuda::std::pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

#if TEST_STD_VER > 2011 // We have the four iteration version
  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb))
         == (cuda::std::pair<II, II>(II(ia + 3), II(ib + 3))));

  assert(cuda::std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), RAI(ib + sb))
         == (cuda::std::pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + 2))
         == (cuda::std::pair<II, II>(II(ia + 2), II(ib + 2))));
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2011
  static_assert(test(), "");
#endif

  return 0;
}
