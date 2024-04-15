//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator, class Predicate>
//     constexpr ForwardIterator       // constexpr after C++17
//     partition_point(ForwardIterator first, ForwardIterator last, Predicate pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct is_odd
{
  __host__ __device__ constexpr bool operator()(const int& i) const
  {
    return i & 1;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  {
    const int ia[] = {2, 4, 6, 8, 10};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia));
  }
  {
    const int ia[] = {1, 2, 4, 6, 8};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 1));
  }
  {
    const int ia[] = {1, 3, 2, 4, 6};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 2));
  }
  {
    const int ia[] = {1, 3, 5, 2, 4, 6};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 3));
  }
  {
    const int ia[] = {1, 3, 5, 7, 2, 4};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 4));
  }
  {
    const int ia[] = {1, 3, 5, 7, 9, 2};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 5));
  }
  {
    const int ia[] = {1, 3, 5, 7, 9, 11};
    assert(
      cuda::std::partition_point(
        forward_iterator<const int*>(cuda::std::begin(ia)), forward_iterator<const int*>(cuda::std::end(ia)), is_odd())
      == forward_iterator<const int*>(ia + 6));
  }
  {
    const int ia[] = {1, 3, 5, 2, 4, 6, 7};
    assert(cuda::std::partition_point(forward_iterator<const int*>(cuda::std::begin(ia)),
                                      forward_iterator<const int*>(cuda::std::begin(ia)),
                                      is_odd())
           == forward_iterator<const int*>(ia));
  }

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
