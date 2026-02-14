//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto begin() requires (!simple-view<V>);
// constexpr auto begin() const requires range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

struct NonCommonSimpleView : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ sentinel_wrapper<int*> end() const;
  __host__ __device__ size_t size()
  {
    return 0;
  } // deliberately non-const
};
static_assert(cuda::std::ranges::sized_range<NonCommonSimpleView>);
static_assert(!cuda::std::ranges::sized_range<const NonCommonSimpleView>);

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // sized_range && random_access_iterator
  {
    cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), RandomAccessIter>);
  }

  {
    const cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), RandomAccessIter>);
  }

  // sized_range && !random_access_iterator
  {
    cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.begin() == cuda::std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), cuda::std::counted_iterator<ForwardIter>>);
  }

  {
    const cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.begin() == cuda::std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), cuda::std::counted_iterator<ForwardIter>>);
  }

  // !sized_range
  {
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == cuda::std::counted_iterator<int*>(buffer, 4));
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), cuda::std::counted_iterator<int*>>);
  }

  {
    const cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == cuda::std::counted_iterator<int*>(buffer, 4));
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), cuda::std::counted_iterator<int*>>);
  }

  // __simple_view<V> && sized_range<V> && !size_range<!V>
  {
    cuda::std::ranges::take_view<NonCommonSimpleView> tv{};
    static_assert(cuda::std::is_same_v<decltype(tv.begin()), cuda::std::counted_iterator<int*>>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::as_const(tv).begin()), cuda::std::counted_iterator<int*>>);
    unused(tv);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
