//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, EquivalenceRelation<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter  // constexpr after C++17
//   adjacent_find(Iter first, Iter last, Pred pred);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(cuda::std::adjacent_find(
           forward_iterator<const int*>(ia), forward_iterator<const int*>(ia + sa), cuda::std::equal_to<int>())
         == forward_iterator<const int*>(ia + 2));
  assert(cuda::std::adjacent_find(
           forward_iterator<const int*>(ia), forward_iterator<const int*>(ia), cuda::std::equal_to<int>())
         == forward_iterator<const int*>(ia));
  assert(cuda::std::adjacent_find(
           forward_iterator<const int*>(ia + 3), forward_iterator<const int*>(ia + sa), cuda::std::equal_to<int>())
         == forward_iterator<const int*>(ia + sa));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
