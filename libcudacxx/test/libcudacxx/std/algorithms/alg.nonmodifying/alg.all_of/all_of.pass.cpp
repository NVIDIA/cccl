//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class Predicate>
//     constexpr bool       // constexpr after C++17
//   all_of(InputIterator first, InputIterator last, Predicate pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct test1
{
  __host__ __device__ constexpr bool operator()(const int& i) const
  {
    return i % 2 == 0;
  }
};

#if TEST_STD_VER > 2011
__host__ __device__ constexpr bool test_constexpr()
{
  int ia[] = {2, 4, 6, 8};
  int ib[] = {2, 4, 5, 8};
  return cuda::std::all_of(cuda::std::begin(ia), cuda::std::end(ia), test1())
      && !cuda::std::all_of(cuda::std::begin(ib), cuda::std::end(ib), test1());
}
#endif

int main(int, char**)
{
  {
    int ia[]              = {2, 4, 6, 8};
    constexpr unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::all_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), test1())
           == true);
    assert(cuda::std::all_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), test1())
           == true);
  }
  {
    constexpr int ia[]    = {2, 4, 5, 8};
    constexpr unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::all_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), test1())
           == false);
    assert(cuda::std::all_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), test1())
           == true);
  }

#if TEST_STD_VER > 2011
  static_assert(test_constexpr(), "");
#endif

  return 0;
}
