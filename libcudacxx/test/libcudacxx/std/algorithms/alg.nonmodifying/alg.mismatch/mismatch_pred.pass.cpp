//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred);
//
// template<InputIterator Iter1, InputIterator Iter2, Predicate Pred>
//   constexpr pair<Iter1, Iter2>   // constexpr after c++17
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred); // C++14

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "counting_predicates.h"
#include "test_iterators.h"
#include "test_macros.h"

#ifdef TEST_COMPILER_MSVC
#  pragma warning(disable : 4018) // signed/unsigned mismatch
#endif // TEST_COMPILER_MSVC

#ifdef TEST_COMPILER_GCC
#  pragma GCC diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_GCC

#ifdef TEST_COMPILER_CLANG
#  pragma clang diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_CLANG

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[]          = {0, 1, 2, 3, 0, 1, 2, 3};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  ((void) sb); // unused in C++11

  typedef cpp17_input_iterator<const int*> II;
  typedef random_access_iterator<const int*> RAI;
  typedef cuda::std::equal_to<int> EQ;

  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib), EQ()) == (cuda::std::pair<II, II>(II(ia + 3), II(ib + 3))));
  assert(cuda::std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), EQ())
         == (cuda::std::pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

  int counter = 0;
  counting_predicate<EQ> bcp(EQ(), counter);
  assert(cuda::std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), bcp)
         == (cuda::std::pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));
  assert(counter > 0 && counter < sa);
  counter = 0;

#if TEST_STD_VER >= 2014
  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb), EQ())
         == (cuda::std::pair<II, II>(II(ia + 3), II(ib + 3))));
  assert(cuda::std::mismatch(RAI(ia), RAI(ia + sa), RAI(ib), RAI(ib + sb), EQ())
         == (cuda::std::pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

  assert(cuda::std::mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb), bcp)
         == (cuda::std::pair<II, II>(II(ia + 3), II(ib + 3))));
  assert(counter > 0 && counter < cuda::std::min(sa, sb));
#endif

  assert(cuda::std::mismatch(ia, ia + sa, ib, EQ()) == (cuda::std::pair<int*, int*>(ia + 3, ib + 3)));

#if TEST_STD_VER >= 2014
  assert(cuda::std::mismatch(ia, ia + sa, ib, ib + sb, EQ()) == (cuda::std::pair<int*, int*>(ia + 3, ib + 3)));
  assert(cuda::std::mismatch(ia, ia + sa, ib, ib + 2, EQ()) == (cuda::std::pair<int*, int*>(ia + 2, ib + 2)));
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}
