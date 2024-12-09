//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// friend constexpr iter_difference_t<W> operator-(const iterator& x, const sentinel& y)
//   requires sized_sentinel_for<Bound, W>;
// friend constexpr iter_difference_t<W> operator-(const sentinel& x, const iterator& y)
//   requires sized_sentinel_for<Bound, W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T>
concept MinusInvocable = requires(cuda::std::ranges::iota_view<T, IntSentinelWith<T>> io) { io.end() - io.begin(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool MinusInvocable = false;

template <class T>
inline constexpr bool MinusInvocable<
  T,
  cuda::std::void_t<decltype(cuda::std::declval<cuda::std::ranges::iota_view<T, IntSentinelWith<T>>>().end()
                             - cuda::std::declval<cuda::std::ranges::iota_view<T, IntSentinelWith<T>>>().begin())>> =
  true;
#endif // TEST_STD_VER <= 2017

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
