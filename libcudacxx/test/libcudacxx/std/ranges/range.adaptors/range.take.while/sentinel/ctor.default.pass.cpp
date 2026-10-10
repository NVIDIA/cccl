//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

struct Sent
{
  bool b; // deliberately uninitialised

  TEST_FUNC friend constexpr bool operator==(int*, const Sent& s)
  {
    return s.b;
  }
#if TEST_STD_VER < 2020
  TEST_FUNC friend constexpr bool operator==(const Sent& s, int*)
  {
    return s.b;
  }
  TEST_FUNC friend constexpr bool operator!=(int*, const Sent& s)
  {
    return !s.b;
  }
  TEST_FUNC friend constexpr bool operator!=(const Sent& s, int*)
  {
    return !s.b;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const
  {
    return nullptr;
  }
  TEST_FUNC Sent end()
  {
    return Sent{};
  }
};

TEST_FUNC constexpr bool test()
{
  {
    using R        = cuda::std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    Sentinel s1 = {};
    assert(!s1.base().b);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
