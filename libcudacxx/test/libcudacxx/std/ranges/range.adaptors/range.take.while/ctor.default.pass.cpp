//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// take_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <bool defaultInitable>
struct View : cuda::std::ranges::view_base
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  TEST_FUNC constexpr explicit View() noexcept {};
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

template <bool defaultInitable>
struct Pred
{
  int i = 0;
  template <bool defaultInitable2 = defaultInitable, cuda::std::enable_if_t<defaultInitable2, int> = 0>
  TEST_FUNC constexpr explicit Pred() noexcept {};
  TEST_FUNC bool operator()(int) const;
};

// clang-format off
static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<true >>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<true >, Pred<false>>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_while_view<View<false>, Pred<false>>>);
// clang-format on

TEST_FUNC constexpr bool test()
{
  {
    cuda::std::ranges::take_while_view<View<true>, Pred<true>> twv = {};
    assert(twv.base().i == 0);
    assert(twv.pred().i == 0);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
