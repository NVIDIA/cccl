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

// friend constexpr iterator operator+(iterator i, difference_type n)
//   requires advanceable<W>;
// friend constexpr iterator operator+(difference_type n, iterator i)
//   requires advanceable<W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // When "_Start" is signed integer like.
  {
    cuda::std::ranges::iota_view<int> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);
  }

  // When "_Start" is not integer like.
  {
    static_assert(cuda::std::totally_ordered<SomeInt>);
    cuda::std::ranges::iota_view io(SomeInt(0));
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);
  }

  // When "_Start" is unsigned integer like and n is greater than or equal to zero.
  {
    cuda::std::ranges::iota_view<unsigned> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);
  }
  {
    cuda::std::ranges::iota_view<unsigned> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(iter1 + 0 == iter2);
  }

  // When "_Start" is unsigned integer like and n is less than zero.
  {
    cuda::std::ranges::iota_view<unsigned> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
