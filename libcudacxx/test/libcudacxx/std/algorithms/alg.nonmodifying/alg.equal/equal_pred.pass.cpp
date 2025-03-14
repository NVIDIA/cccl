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
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred);
//
// Introduced in C++14:
// template<InputIterator Iter1, InputIterator Iter2,
//          Predicate<auto, Iter1::value_type, Iter2::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "counting_predicates.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int ia[]         = {0, 1, 2, 3, 4, 5};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  int ib[s]        = {0, 1, 2, 5, 4, 5};
  assert(cuda::std::equal(
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cpp17_input_iterator<const int*>(ia),
    cuda::std::equal_to<int>()));
  assert(cuda::std::equal(
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cuda::std::equal_to<int>()));
  assert(cuda::std::equal(
    random_access_iterator<const int*>(ia),
    random_access_iterator<const int*>(ia + s),
    random_access_iterator<const int*>(ia),
    random_access_iterator<const int*>(ia + s),
    cuda::std::equal_to<int>()));

  typedef cuda::std::equal_to<int> EQ;
  int comparison_count = 0;
  counting_predicate<EQ> counting_equals(EQ(), comparison_count);
  assert(!cuda::std::equal(
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s - 1),
    counting_equals));
  assert(comparison_count > 0);
  comparison_count = 0;
  assert(!cuda::std::equal(
    random_access_iterator<const int*>(ia),
    random_access_iterator<const int*>(ia + s),
    random_access_iterator<const int*>(ia),
    random_access_iterator<const int*>(ia + s - 1),
    counting_equals));
  assert(comparison_count == 0);

  assert(!cuda::std::equal(
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cpp17_input_iterator<const int*>(ib),
    cuda::std::equal_to<int>()));

  assert(!cuda::std::equal(
    cpp17_input_iterator<const int*>(ia),
    cpp17_input_iterator<const int*>(ia + s),
    cpp17_input_iterator<const int*>(ib),
    cpp17_input_iterator<const int*>(ib + s),
    cuda::std::equal_to<int>()));
  assert(!cuda::std::equal(
    random_access_iterator<const int*>(ia),
    random_access_iterator<const int*>(ia + s),
    random_access_iterator<const int*>(ib),
    random_access_iterator<const int*>(ib + s),
    cuda::std::equal_to<int>()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
