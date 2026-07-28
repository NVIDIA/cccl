//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr take_while_view(V base, Pred pred);

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
  bool copied = false;
  bool moved  = false;
  Pred()      = default;
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
    cuda::std::ranges::take_while_view<View, Pred> twv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(twv.pred().moved);
    assert(!twv.pred().copied);
    assert(cuda::std::move(twv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
