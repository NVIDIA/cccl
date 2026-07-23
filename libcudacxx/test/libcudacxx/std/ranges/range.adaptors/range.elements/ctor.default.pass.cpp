//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// elements_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <bool DefaultInitializable>
struct View : cuda::std::ranges::view_base
{
  int i = 42;
  template <bool DefaultInitializable2 = DefaultInitializable, cuda::std::enable_if_t<DefaultInitializable2, int> = 0>
  TEST_FUNC constexpr explicit View() noexcept
  {}
  TEST_FUNC cuda::std::tuple<int>* begin() const;
  TEST_FUNC cuda::std::tuple<int>* end() const;
};

static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::elements_view<View<true>, 0>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::elements_view<View<false>, 0>>);

TEST_FUNC constexpr bool test()
{
  {
    cuda::std::ranges::elements_view<View<true>, 0> ev = {};
    assert(ev.base().i == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
