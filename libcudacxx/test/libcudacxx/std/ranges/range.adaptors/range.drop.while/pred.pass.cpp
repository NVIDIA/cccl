//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const Pred& pred() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct View : cuda::std::ranges::view_interface<View>
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct Pred
{
  int i;
  TEST_FUNC bool operator()(int) const;
};

TEST_FUNC constexpr bool test()
{
  // &
  {
    cuda::std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = dwv.pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &
  {
    const cuda::std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = dwv.pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // &&
  {
    cuda::std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = cuda::std::move(dwv).pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  // const &&
  {
    const cuda::std::ranges::drop_while_view<View, Pred> dwv{{}, Pred{5}};
    decltype(auto) x = cuda::std::move(dwv).pred();
    static_assert(cuda::std::same_as<decltype(x), Pred const&>);
    assert(x.i == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020
  return 0;
}
