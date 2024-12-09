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

// constexpr auto end() requires (!simple-view<V>)
// constexpr auto end() const requires range<const V>

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // sized_range && random_access_iterator
  {
    cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 0);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 0));
    ASSERT_SAME_TYPE(decltype(tv.end()), RandomAccessIter);
  }

  {
    const cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 1);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 1));
    ASSERT_SAME_TYPE(decltype(tv.end()), RandomAccessIter);
  }

  // sized_range && !random_access_iterator
  {
    cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 2);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 2));
    ASSERT_SAME_TYPE(decltype(tv.end()), cuda::std::default_sentinel_t);
  }

  {
    const cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 3);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 3));
    ASSERT_SAME_TYPE(decltype(tv.end()), cuda::std::default_sentinel_t);
  }

  // !sized_range
  {
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 4));

    // The <sentinel> type.
    static_assert(!cuda::std::same_as<decltype(tv.end()), cuda::std::default_sentinel_t>);
    static_assert(!cuda::std::same_as<decltype(tv.end()), int*>);
  }

  {
    const cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 5);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 5));
  }

  // Just to cover the case where count == 8.
  {
    cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 8);
    assert(tv.end() == cuda::std::ranges::next(tv.begin(), 8));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
