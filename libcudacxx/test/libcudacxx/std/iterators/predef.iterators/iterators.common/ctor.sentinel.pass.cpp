//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

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
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
  {
    assert(lv == cuda::std::next(CommonIt(it)));
  }
#endif

  CommonIt rv = CommonIt(cuda::std::move(sent));
  assert(rv == CommonIt(sent));
  assert(rv != CommonIt(it));
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
  {
    assert(rv == cuda::std::next(CommonIt(it)));
  }
#endif

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
