//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Range, class Pred>
// filter_view(Range&&, Pred) -> filter_view<views::all_t<Range>, Pred>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  constexpr View() = default;
  TEST_FUNC forward_iterator<int*> begin() const;
  TEST_FUNC sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(cuda::std::ranges::view<View>);

// A range that is not a view
struct Range
{
  constexpr Range() = default;
  TEST_FUNC forward_iterator<int*> begin() const;
  TEST_FUNC sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(cuda::std::ranges::range<Range> && !cuda::std::ranges::view<Range>);

struct Pred
{
  TEST_FUNC constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    View v{};
    Pred pred{};
    cuda::std::ranges::filter_view view(v, pred);
    static_assert(cuda::std::is_same_v<decltype(view), cuda::std::ranges::filter_view<View, Pred>>);
  }

  {
    // Test with a range that isn't a view, to make sure we properly use views::all_t in the
    // implementation.

    Range r{};
    Pred pred{};
    cuda::std::ranges::filter_view view(r, pred);
    static_assert(
      cuda::std::is_same_v<decltype(view), cuda::std::ranges::filter_view<cuda::std::ranges::ref_view<Range>, Pred>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
