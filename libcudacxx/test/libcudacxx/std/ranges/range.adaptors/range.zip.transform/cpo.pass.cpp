//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// cuda::std::views::zip_transform

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/vector>

#include "types.h"

struct NotMoveConstructible
{
  NotMoveConstructible()                       = default;
  NotMoveConstructible(NotMoveConstructible&&) = delete;
  TEST_FUNC int operator()() const
  {
    return 5;
  }
};

struct NotCopyConstructible
{
  NotCopyConstructible()                            = default;
  NotCopyConstructible(NotCopyConstructible&&)      = default;
  NotCopyConstructible(const NotCopyConstructible&) = delete;
  TEST_FUNC int operator()() const
  {
    return 5;
  }
};

struct NotInvocable
{};

template <class... Args>
struct Invocable
{
  TEST_FUNC int operator()(Args...) const
  {
    return 5;
  }
};

struct ReturnNotObject
{
  TEST_FUNC void operator()() const {}
};

// LWG3773 views::zip_transform still requires F to be copy_constructible when empty pack
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), NotCopyConstructible>);

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform))>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), NotMoveConstructible>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), NotInvocable>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), Invocable<>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), ReturnNotObject>);

static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), //
                                        Invocable<int>, //
                                        cuda::std::ranges::iota_view<int, int>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)), //
                                         Invocable<>, //
                                         cuda::std::ranges::iota_view<int, int>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)),
                                         Invocable<int>,
                                         cuda::std::ranges::iota_view<int, int>,
                                         cuda::std::ranges::iota_view<int, int>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::zip_transform)),
                                        Invocable<int, int>,
                                        cuda::std::ranges::iota_view<int, int>,
                                        cuda::std::ranges::iota_view<int, int>>);

TEST_FUNC constexpr bool test()
{
  {
    // zip_transform function with no ranges
    auto v = cuda::std::views::zip_transform(Invocable<>{});
    assert(cuda::std::ranges::empty(v));
    static_assert(cuda::std::is_same_v<decltype(v), cuda::std::ranges::empty_view<int>>);
  }

  {
    // zip_transform views
    int buffer1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int buffer2[] = {9, 10, 11, 12};
    auto view1    = cuda::std::views::all(buffer1);
    auto view2    = cuda::std::views::all(buffer2);
    cuda::std::same_as<
      cuda::std::ranges::zip_transform_view<cuda::std::plus<>, decltype(view1), decltype(view2)>> decltype(auto) v =
      cuda::std::views::zip_transform(cuda::std::plus{}, buffer1, buffer2);
    assert(cuda::std::ranges::size(v) == 4);
    auto expected = {10, 12, 14, 16};
    assert(cuda::std::ranges::equal(v, expected));
    static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<decltype(v)>, int>);
  }

  {
    // zip_transform a viewable range
    cuda::std::array a{1, 2, 3};
    auto id = [](auto& x) -> decltype(auto) {
      return (x);
    };
    cuda::std::same_as<
      cuda::std::ranges::zip_transform_view<decltype(id),
                                            cuda::std::ranges::ref_view<cuda::std::array<int, 3>>>> decltype(auto) v =
      cuda::std::views::zip_transform(id, a);
    assert(&v[0] == &a[0]);
    static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<decltype(v)>, int&>);
  }

  int buffer[] = {1, 2, 3};
  {
    // one range
    auto v = cuda::std::views::zip_transform(MakeTuple{}, SimpleCommon{buffer});
    assert(
      cuda::std::ranges::equal(v, cuda::std::vector{cuda::std::tuple(1), cuda::std::tuple(2), cuda::std::tuple(3)}));
  }

  {
    // two ranges
    auto v = cuda::std::views::zip_transform(GetFirst{}, SimpleCommon{buffer}, cuda::std::views::iota(0));
    assert(cuda::std::ranges::equal(v, cuda::std::vector{1, 2, 3}));
  }

  {
    // three ranges
    auto v = cuda::std::views::zip_transform(
      Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::single_view(2.));
    assert(cuda::std::ranges::equal(v, cuda::std::vector{cuda::std::tuple(1, 1, 2.0)}));
  }

  {
    // single empty range
    auto v = cuda::std::views::zip_transform(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range at the beginning
    auto v = cuda::std::views::zip_transform(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range in the middle
    auto v = cuda::std::views::zip_transform(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range at the end
    auto v = cuda::std::views::zip_transform(
      MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>());
    assert(cuda::std::ranges::empty(v));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
