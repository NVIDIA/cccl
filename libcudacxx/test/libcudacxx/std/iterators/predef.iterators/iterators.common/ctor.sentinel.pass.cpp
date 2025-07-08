//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr common_iterator(S s);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr bool test()
{
  using Sent     = sentinel_wrapper<It>;
  using CommonIt = cuda::std::common_iterator<It, Sent>;
  int a[]        = {1, 2, 3};
  It it          = It(a);
  Sent sent      = Sent(It(a + 1));

  CommonIt lv = CommonIt(sent);
  assert(lv == CommonIt(sent));
  assert(lv != CommonIt(it));
  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    assert(lv == cuda::std::next(CommonIt(it)));
  }

  CommonIt rv = CommonIt(cuda::std::move(sent));
  assert(rv == CommonIt(sent));
  assert(rv != CommonIt(it));
  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    assert(rv == cuda::std::next(CommonIt(it)));
  }

  return true;
}

int main(int, char**)
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<const int*>();

  static_assert(test<cpp17_input_iterator<int*>>());
  static_assert(test<forward_iterator<int*>>());
  static_assert(test<bidirectional_iterator<int*>>());
  static_assert(test<random_access_iterator<int*>>());
  static_assert(test<contiguous_iterator<int*>>());
  static_assert(test<int*>());
  static_assert(test<const int*>());

  return 0;
}
