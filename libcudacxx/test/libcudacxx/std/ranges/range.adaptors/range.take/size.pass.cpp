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

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

#if TEST_STD_VER > 2017
template <class T>
concept SizeEnabled = requires(const cuda::std::ranges::take_view<T>& tv) { tv.size(); };
#else
template <class T, class = void>
inline constexpr bool SizeEnabled = false;

template <class T>
inline constexpr bool
  SizeEnabled<T, cuda::std::void_t<decltype(cuda::std::declval<const cuda::std::ranges::take_view<T>&>().size())>> =
    true;
#endif

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(SizeEnabled<SizedRandomAccessView>);
    static_assert(!SizeEnabled<CopyableView>);
  }

  {
    cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 0);
    assert(tv.size() == 0);
  }

  {
    const cuda::std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 2);
    assert(tv.size() == 2);
  }

  {
    cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.size() == 4);
  }

  {
    const cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 6);
    assert(tv.size() == 6);
  }

  {
    cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 8);
    assert(tv.size() == 8);
  }
  {
    const cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 8);
    assert(tv.size() == 8);
  }

  {
    cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 10);
    assert(tv.size() == 8);
  }
  {
    const cuda::std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 10);
    assert(tv.size() == 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
