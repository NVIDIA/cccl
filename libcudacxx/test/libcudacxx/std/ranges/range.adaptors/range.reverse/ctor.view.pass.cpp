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

// constexpr explicit reverse_view(V r);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    BidirRange r{buffer, buffer + 8};
    cuda::std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    const BidirRange r{buffer, buffer + 8};
    const cuda::std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    cuda::std::ranges::reverse_view<BidirSentRange<MoveOnly>> rev(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    auto moved = cuda::std::move(rev).base();
    assert(moved.begin_ == buffer);
    assert(moved.end_ == buffer + 8);
  }
  {
    const cuda::std::ranges::reverse_view<BidirSentRange<Copyable>> rev(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    // Make sure this ctor is marked as "explicit".
    static_assert(cuda::std::is_constructible_v<cuda::std::ranges::reverse_view<BidirRange>, BidirRange>);
    static_assert(!cuda::std::is_convertible_v<cuda::std::ranges::reverse_view<BidirRange>, BidirRange>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test(), "");
#endif

  return 0;
}
