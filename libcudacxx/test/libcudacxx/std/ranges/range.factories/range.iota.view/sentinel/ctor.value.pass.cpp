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

// constexpr explicit sentinel(Bound bound);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    using Sent = cuda::std::ranges::sentinel_t<cuda::std::ranges::iota_view<int, IntSentinelWith<int>>>;
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::iota_view<int, IntSentinelWith<int>>>;
    auto sent  = Sent(IntSentinelWith<int>(42));
    assert(sent == Iter(42));
  }
  {
    using Sent = cuda::std::ranges::sentinel_t<cuda::std::ranges::iota_view<SomeInt, IntSentinelWith<SomeInt>>>;
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::iota_view<SomeInt, IntSentinelWith<SomeInt>>>;
    auto sent  = Sent(IntSentinelWith<SomeInt>(SomeInt(42)));
    assert(sent == Iter(SomeInt(42)));
  }
  {
    using Sent = cuda::std::ranges::sentinel_t<cuda::std::ranges::iota_view<SomeInt, IntSentinelWith<SomeInt>>>;
    static_assert(!cuda::std::is_convertible_v<Sent, IntSentinelWith<SomeInt>>);
    static_assert(cuda::std::is_constructible_v<Sent, IntSentinelWith<SomeInt>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
