//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_difference_t<W> operator-(const iterator& x, const sentinel& y)
//   requires sized_sentinel_for<Bound, W>;
// friend constexpr iter_difference_t<W> operator-(const sentinel& x, const iterator& y)
//   requires sized_sentinel_for<Bound, W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT MinusInvocable =
  _CCCL_REQUIRES_EXPR((T), cuda::std::ranges::iota_view<T, IntSentinelWith<T>> io)(io.end() - io.begin());

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto outIter = random_access_iterator<int*>(buffer);
    cuda::std::ranges::iota_view<random_access_iterator<int*>, IntSentinelWith<random_access_iterator<int*>>> io(
      outIter, IntSentinelWith<random_access_iterator<int*>>(cuda::std::ranges::next(outIter, 8)));
    auto iter = io.begin();
    auto sent = io.end();
    assert(iter - sent == -8);
    assert(sent - iter == 8);
  }
  {
    auto outIter = random_access_iterator<int*>(buffer);
    const cuda::std::ranges::iota_view<random_access_iterator<int*>, IntSentinelWith<random_access_iterator<int*>>> io(
      outIter, IntSentinelWith<random_access_iterator<int*>>(cuda::std::ranges::next(outIter, 8)));
    const auto iter = io.begin();
    const auto sent = io.end();
    assert(iter - sent == -8);
    assert(sent - iter == 8);
  }

  {
    // The minus operator requires that "W" is an input_or_output_iterator.
    static_assert(!MinusInvocable<int>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
