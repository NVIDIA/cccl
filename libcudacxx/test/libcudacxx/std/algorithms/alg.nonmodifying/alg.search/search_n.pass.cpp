//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator, class Size, class T>
//   constexpr ForwardIterator     // constexpr after C++17
//   search_n(ForwardIterator first, ForwardIterator last, Size count,
//            const T& value);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

#ifdef TEST_COMPILER_MSVC
#  pragma warning(disable : 4018) // signed/unsigned mismatch
#endif // TEST_COMPILER_MSVC

#ifdef TEST_COMPILER_GCC
#  pragma GCC diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_GCC

#ifdef TEST_COMPILER_CLANG
#  pragma clang diagnostic ignored "-Wsign-compare"
#endif // TEST_COMPILER_CLANG

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]          = {0, 1, 2, 3, 4, 5};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 0) == Iter(ia));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 0) == Iter(ia + 0));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 0) == Iter(ia + sa));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 0) == Iter(ia + sa));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 3) == Iter(ia));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 3) == Iter(ia + 3));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 3) == Iter(ia + sa));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 3) == Iter(ia + sa));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 0, 5) == Iter(ia));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 1, 5) == Iter(ia + 5));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), 2, 5) == Iter(ia + sa));
  assert(cuda::std::search_n(Iter(ia), Iter(ia + sa), sa, 5) == Iter(ia + sa));

  int ib[]          = {0, 0, 1, 1, 2, 2};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 0) == Iter(ib));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 0) == Iter(ib + 0));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 0) == Iter(ib + 0));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 0) == Iter(ib + sb));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 0) == Iter(ib + sb));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 1) == Iter(ib));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 1) == Iter(ib + 2));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 1) == Iter(ib + 2));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 1) == Iter(ib + sb));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 1) == Iter(ib + sb));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 0, 2) == Iter(ib));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 1, 2) == Iter(ib + 4));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 2, 2) == Iter(ib + 4));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), 3, 2) == Iter(ib + sb));
  assert(cuda::std::search_n(Iter(ib), Iter(ib + sb), sb, 2) == Iter(ib + sb));

  int ic[]          = {0, 0, 0};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 0, 0) == Iter(ic));
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 1, 0) == Iter(ic));
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 2, 0) == Iter(ic));
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 3, 0) == Iter(ic));
  assert(cuda::std::search_n(Iter(ic), Iter(ic + sc), 4, 0) == Iter(ic + sc));

  // Check that we properly convert the size argument to an integral.
  (void) cuda::std::search_n(Iter(ic), Iter(ic + sc), UserDefinedIntegral<unsigned>(0), 0);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
