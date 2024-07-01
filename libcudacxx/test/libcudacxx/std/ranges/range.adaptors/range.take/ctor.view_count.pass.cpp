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

// constexpr take_view(V base, range_difference_t<V> count);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 0);
    assert(tv.base().ptr_ == buffer);
    assert(tv.begin() == tv.end()); // Checking we have correct size.
  }

  {
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 1);
    assert(cuda::std::move(tv).base().ptr_ == buffer);
    assert(cuda::std::ranges::next(tv.begin(), 1) == tv.end()); // Checking we have correct size.
  }

  {
    const cuda::std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 2);
    assert(tv.base().ptr_ == buffer);
    assert(cuda::std::ranges::next(tv.begin(), 2) == tv.end()); // Checking we have correct size.
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
