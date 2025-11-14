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
//   bool
//   none_of(InputIterator first, InputIterator last, Predicate pred);

#include <cuda/std/algorithm>
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

__host__ __device__ constexpr bool test_constexpr()
{
  int ia[] = {1, 3, 6, 7};
  int ib[] = {1, 3, 5, 7};
  return !cuda::std::none_of(cuda::std::begin(ia), cuda::std::end(ia), test1())
      && cuda::std::none_of(cuda::std::begin(ib), cuda::std::end(ib), test1());
}

int main(int, char**)
{
  {
    int ia[]          = {2, 4, 6, 8};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), test1())
           == false);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), test1())
           == true);
  }
  {
    const int ia[]    = {2, 4, 5, 8};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), test1())
           == false);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), test1())
           == true);
  }
  {
    const int ia[]    = {1, 3, 5, 7};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia + sa), test1())
           == true);
    assert(cuda::std::none_of(cpp17_input_iterator<const int*>(ia), cpp17_input_iterator<const int*>(ia), test1())
           == true);
  }

  static_assert(test_constexpr(), "");

  return 0;
}
