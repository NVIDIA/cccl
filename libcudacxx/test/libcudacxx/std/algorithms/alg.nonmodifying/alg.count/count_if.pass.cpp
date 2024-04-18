//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter::difference_type   // constexpr after C++17
//   count_if(Iter first, Iter last, Pred pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

struct eq
{
  __host__ __device__ constexpr eq(int val)
      : v(val)
  {}
  __host__ __device__ constexpr bool operator()(int v2) const
  {
    return v == v2;
  }
  int v;
};

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  int ia[]          = {0, 1, 2, 2, 0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  assert(cuda::std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), eq(2))
         == 3);
  assert(cuda::std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), eq(7))
         == 0);
  assert(cuda::std::count_if(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), eq(2)) == 0);

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
