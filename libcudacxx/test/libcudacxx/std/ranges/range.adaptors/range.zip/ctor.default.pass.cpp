//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// zip_view() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int buff[] = {1, 2, 3};

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr DefaultConstructibleView()
      : begin_(buff)
      , end_(buff + 3)
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

struct NoDefaultCtrView : cuda::std::ranges::view_base
{
  NoDefaultCtrView() = delete;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the tuple's constructor. If any of the iterators are
// not default constructible, zip iterator's =default would be implicitly deleted.
static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<DefaultConstructibleView>>);
static_assert(cuda::std::is_default_constructible_v<
              cuda::std::ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>>);
static_assert(
  !cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<DefaultConstructibleView, NoDefaultCtrView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<NoDefaultCtrView, NoDefaultCtrView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<NoDefaultCtrView>>);

TEST_FUNC constexpr bool test()
{
  {
    using View = cuda::std::ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>;
    View v     = View(); // the default constructor is not explicit
    assert(v.size() == 3);
    auto it    = v.begin();
    using Pair = cuda::std::tuple<const int&, const int&>;
    assert(*it++ == Pair(buff[0], buff[0]));
    assert(*it++ == Pair(buff[1], buff[1]));
    assert(*it == Pair(buff[2], buff[2]));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
