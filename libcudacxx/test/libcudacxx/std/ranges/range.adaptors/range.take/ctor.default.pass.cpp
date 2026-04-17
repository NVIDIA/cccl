//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// take_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int buff[8] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructible : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr DefaultConstructible()
      : begin_(buff)
      , end_(buff + 8)
  {}
  TEST_FUNC constexpr int const* begin() const
  {
    return begin_;
  }
  TEST_FUNC constexpr int const* end() const
  {
    return end_;
  }

private:
  int const* begin_;
  int const* end_;
};

struct NonDefaultConstructible : cuda::std::ranges::view_base
{
  NonDefaultConstructible() = delete;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

TEST_FUNC constexpr bool test()
{
  {
    cuda::std::ranges::take_view<DefaultConstructible> tv;
    assert(tv.begin() == buff);
    assert(tv.size() == 0);
  }

  // Test SFINAE-friendliness
  {
    static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<DefaultConstructible>>);
    static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<NonDefaultConstructible>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
