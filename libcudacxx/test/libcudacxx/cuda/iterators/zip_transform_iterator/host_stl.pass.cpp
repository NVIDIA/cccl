//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include <iterator>

#include "test_iterators.h"
#include "test_macros.h"

struct plus_1337
{
  constexpr int operator()(const int& value) const noexcept
  {
    return value + 42;
  }
};

constexpr bool test()
{
  int data[100] = {};
  cuda::zip_transform_iterator<plus_1337, advance_only_iterator> iter1{plus_1337{}, advance_only_iterator{data}};
  cuda::zip_transform_iterator<plus_1337, advance_only_iterator> iter2{plus_1337{}, advance_only_iterator{data + 42}};
  const int dist = 42;

  {
    auto diff = ::std::distance(iter1, iter2);
    static_assert(
      cuda::std::is_same_v<decltype(diff),
                           cuda::std::iter_difference_t<cuda::zip_transform_iterator<plus_1337, advance_only_iterator>>>);
    assert(diff == dist);
  }

  {
    auto iter = ::std::next(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_transform_iterator<plus_1337, advance_only_iterator>>);
    assert(iter == iter2);
  }

  {
    auto iter = ::std::prev(iter2, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::zip_transform_iterator<plus_1337, advance_only_iterator>>);
    assert(iter == iter1);
  }

  {
    ::std::advance(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(::std::advance(iter1, dist)), void>);
    assert(iter1 == iter2);
  }

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 test();
                 static_assert(test());
               }))

  return 0;
}
