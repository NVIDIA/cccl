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
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView(buffer), 4);
    decltype(auto) sw1 = tv.end().base();
    static_assert(cuda::std::same_as<decltype(sw1), sentinel_wrapper<int*>>);
    assert(base(sw1) == buffer + 8);
    decltype(auto) sw2 = cuda::std::as_const(tv).end().base();
    static_assert(cuda::std::same_as<decltype(sw2), sentinel_wrapper<int*>>);
    assert(base(sw2) == buffer + 8);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
