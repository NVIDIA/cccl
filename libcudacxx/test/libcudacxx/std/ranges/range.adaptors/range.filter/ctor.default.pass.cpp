//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// filter_view() requires cuda::std::default_initializable<View> &&
//                        cuda::std::default_initializable<Pred> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

_CCCL_GLOBAL_CONSTANT int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr DefaultConstructibleView()
      : begin_(buff)
      , end_(buff + 8)
  {}
  TEST_FUNC constexpr const int* begin() const
  {
    return begin_;
  }
  TEST_FUNC constexpr const int* end() const
  {
    return end_;
  }

private:
  int const* begin_{};
  int const* end_{};
};

struct DefaultConstructiblePredicate
{
  DefaultConstructiblePredicate() = default;

  TEST_FUNC constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

struct NoDefaultView : cuda::std::ranges::view_base
{
  NoDefaultView() = delete;
  TEST_FUNC int* begin() const
  {
    return nullptr;
  }
  TEST_FUNC int* end() const
  {
    return nullptr;
  }
};

struct NoDefaultPredicate
{
  NoDefaultPredicate() = delete;
  TEST_FUNC constexpr bool operator()(int) const
  {
    return true;
  }
};

struct NoexceptView : cuda::std::ranges::view_base
{
  constexpr NoexceptView() noexcept = default;
  TEST_FUNC int const* begin() const
  {
    return nullptr;
  }
  TEST_FUNC int const* end() const
  {
    return nullptr;
  }
};

struct NoexceptPredicate
{
  constexpr NoexceptPredicate() noexcept = default;
  TEST_FUNC bool operator()(int) const
  {
    return true;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;

    View view;
    auto it  = view.begin();
    auto end = view.end();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it++ == 6);
    assert(*it++ == 8);
    assert(it == end);
  }

  {
    using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;

    View view = {};
    auto it = view.begin(), end = view.end();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it++ == 6);
    assert(*it++ == 8);
    assert(it == end);
  }

  // Check cases where the default constructor isn't provided
  {
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::filter_view<DefaultConstructibleView, NoDefaultPredicate>>);

    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::filter_view<NoDefaultView, DefaultConstructiblePredicate>>);
    static_assert(
      !cuda::std::is_default_constructible_v<cuda::std::ranges::filter_view<NoDefaultView, NoDefaultPredicate>>);
    //     || TEST_COMPILER(MSVC, <, 19, 44)) && (TEST_STD_VER == 2020))
  }

  // Check noexcept-ness
  {
    {
      using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
      // GCC 7 simply gives the wrong answer here. No amount of cajoling, pleading, or
      // massaging the code ever got it to pass this static_assert()
#if !TEST_COMPILER(GCC)
      static_assert(!cuda::std::is_nothrow_default_constructible_v<View>);
#endif // !TEST_COMPILER(GCC)
    }
    {
      using View = cuda::std::ranges::filter_view<NoexceptView, NoexceptPredicate>;
      static_assert(cuda::std::is_nothrow_default_constructible_v<View>);
    }
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
