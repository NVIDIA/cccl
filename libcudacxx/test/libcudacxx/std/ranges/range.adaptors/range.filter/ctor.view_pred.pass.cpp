//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr filter_view(View, Pred); // explicit since C++23

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_convertible.h"
#include "test_macros.h"
#include "types.h"

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  [[nodiscard]] TEST_FUNC constexpr int* begin() const
  {
    return begin_;
  }
  [[nodiscard]] TEST_FUNC constexpr int* end() const
  {
    return end_;
  }

private:
  int* begin_;
  int* end_;
};

struct Pred
{
  [[nodiscard]] TEST_FUNC constexpr bool operator()(int i) const
  {
    return i % 2 != 0;
  }
};

struct TrackingPred : TrackInitialization
{
  using TrackInitialization::TrackInitialization;
  TEST_FUNC constexpr bool operator()(int) const
  {
    return true;
  }
};

struct TrackingRange
    : TrackInitialization
    , cuda::std::ranges::view_base
{
  using TrackInitialization::TrackInitialization;
  [[nodiscard]] TEST_FUNC int* begin() const
  {
    return nullptr;
  }
  [[nodiscard]] TEST_FUNC int* end() const
  {
    return nullptr;
  }
};

// SFINAE tests.

static_assert(!test_convertible<cuda::std::ranges::filter_view<Range, Pred>, Range, Pred>(),
              "This constructor must be explicit");

TEST_FUNC constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test explicit syntax
  {
    Range range(buff, buff + 8);
    Pred pred{};
    cuda::std::ranges::filter_view<Range, Pred> view(range, pred);
    auto it = view.begin(), end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 3);
    assert(*it++ == 5);
    assert(*it++ == 7);
    assert(it == end);
  }

  // Make sure we move the view
  {
    bool moved = false, copied = false;
    TrackingRange range(&moved, &copied);
    Pred pred{};
    [[maybe_unused]] cuda::std::ranges::filter_view<TrackingRange, Pred> view(cuda::std::move(range), pred);
    assert(moved);
    assert(!copied);
  }

  // Make sure we move the predicate
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 8);
    TrackingPred pred(&moved, &copied);
    [[maybe_unused]] cuda::std::ranges::filter_view<Range, TrackingPred> view(range, cuda::std::move(pred));
    assert(moved);
    assert(!copied);
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
