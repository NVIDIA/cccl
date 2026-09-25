//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_macros.h"

struct PODSentinel
{
  int i; // deliberately uninitialised

  TEST_FUNC friend constexpr bool operator==(cuda::std::tuple<int>*, const PODSentinel&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  TEST_FUNC friend constexpr bool operator==(const PODSentinel&, cuda::std::tuple<int>*)
  {
    return true;
  }
  TEST_FUNC friend constexpr bool operator!=(cuda::std::tuple<int>*, const PODSentinel&)
  {
    return false;
  }
  TEST_FUNC friend constexpr bool operator!=(const PODSentinel&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC cuda::std::tuple<int>* begin() const
  {
    return nullptr;
  }
  TEST_FUNC PODSentinel end()
  {
    return PODSentinel{};
  }
};

TEST_FUNC constexpr bool test()
{
  using EleView  = cuda::std::ranges::elements_view<Range, 0>;
  using Sentinel = cuda::std::ranges::sentinel_t<EleView>;
  static_assert(!cuda::std::is_same_v<Sentinel, cuda::std::ranges::iterator_t<EleView>>);

  {
    Sentinel s;
    assert(s.base().i == 0);
  }
  {
    Sentinel s = {};
    assert(s.base().i == 0);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
