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

#include "test_macros.h"

constexpr bool test()
{
  cuda::counting_iterator<int> iter1{42};
  cuda::counting_iterator<int> iter2{1337};
  const int dist = 1337 - 42;

  {
    auto diff = ::std::distance(iter1, iter2);
    static_assert(cuda::std::is_same_v<decltype(diff), cuda::std::iter_difference_t<cuda::counting_iterator<int>>>);
    static_assert(noexcept(::std::distance(iter1, iter2)));
    assert(diff == dist);
  }

  {
    auto iter = ::std::next(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::counting_iterator<int>>);
    static_assert(noexcept(::std::next(iter1, dist)));
    assert(iter == iter2);
  }

  {
    auto iter = ::std::prev(iter2, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::counting_iterator<int>>);
    static_assert(noexcept(::std::prev(iter2, dist)));
    assert(iter == iter1);
  }

  {
    ::std::advance(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(::std::advance(iter1, dist)), void>);
    static_assert(noexcept(::std::advance(iter1, dist)));
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
