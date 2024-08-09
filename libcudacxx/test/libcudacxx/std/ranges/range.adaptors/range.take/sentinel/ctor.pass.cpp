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

// sentinel() = default;
// constexpr explicit sentinel(sentinel_t<Base> end);
// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // Test the default ctor.
    using TakeView = cuda::std::ranges::take_view<MoveOnlyView>;
    using Sentinel = cuda::std::ranges::sentinel_t<TakeView>;
    Sentinel s;
    TakeView tv = TakeView(MoveOnlyView(buffer), 4);
    assert(tv.begin() + 4 == s);
  }

  {
    // Test the conversion from "sentinel" to "sentinel-to-const".
    using TakeView      = cuda::std::ranges::take_view<MoveOnlyView>;
    using Sentinel      = cuda::std::ranges::sentinel_t<TakeView>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const TakeView>;
    static_assert(cuda::std::is_convertible_v<Sentinel, ConstSentinel>);
    TakeView tv      = TakeView(MoveOnlyView(buffer), 4);
    Sentinel s       = tv.end();
    ConstSentinel cs = s;
    cs               = s; // test assignment also
    assert(tv.begin() + 4 == s);
    assert(tv.begin() + 4 == cs);
    assert(cuda::std::as_const(tv).begin() + 4 == s);
    assert(cuda::std::as_const(tv).begin() + 4 == cs);
  }

  {
    // Test the constructor from "base-sentinel" to "sentinel".
    using TakeView             = cuda::std::ranges::take_view<MoveOnlyView>;
    using Sentinel             = cuda::std::ranges::sentinel_t<TakeView>;
    sentinel_wrapper<int*> sw1 = MoveOnlyView(buffer).end();
    static_assert(cuda::std::is_constructible_v<Sentinel, sentinel_wrapper<int*>>);
    static_assert(!cuda::std::is_convertible_v<sentinel_wrapper<int*>, Sentinel>);
    auto s             = Sentinel(sw1);
    decltype(auto) sw2 = s.base();
    static_assert(cuda::std::same_as<decltype(sw2), sentinel_wrapper<int*>>);
    assert(base(sw2) == base(sw1));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
