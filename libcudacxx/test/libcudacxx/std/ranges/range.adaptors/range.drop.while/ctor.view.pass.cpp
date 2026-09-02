//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr drop_while_view(V base, Pred pred);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  MoveOnly mo;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct Pred
{
  bool copied      = false;
  bool moved       = false;
  constexpr Pred() = default;
  TEST_FUNC constexpr Pred(Pred&&)
      : moved(true)
  {}
  TEST_FUNC constexpr Pred(const Pred&)
      : copied(true)
  {}
  TEST_FUNC bool operator()(int) const;
};

TEST_FUNC constexpr bool test()
{
  {
    cuda::std::ranges::drop_while_view<View, Pred> dwv{View{{}, MoveOnly{5}}, Pred{}};
    assert(dwv.pred().moved);
    assert(!dwv.pred().copied);
    assert(cuda::std::move(dwv).base().mo.get() == 5);
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
