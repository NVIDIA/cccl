//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit zip_transform_view(F, Views...)

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/vector>

#include "types.h"

struct Fn
{
  template <class... T>
  TEST_FUNC int operator()(T&&...) const
  {
    return 5;
  }
};

template <class T, class... Args>
_CCCL_CONCEPT IsImplicitlyConstructible =
  _CCCL_REQUIRES_EXPR((T, variadic Args), T val, Args... args)(val = {cuda::std::forward<Args>(args)...});

// test constructor is explicit
static_assert(cuda::std::constructible_from<cuda::std::ranges::zip_transform_view<Fn, IntView>, Fn, IntView>);
static_assert(!IsImplicitlyConstructible<cuda::std::ranges::zip_transform_view<Fn, IntView>, Fn, IntView>);

static_assert(
  cuda::std::constructible_from<cuda::std::ranges::zip_transform_view<Fn, IntView, IntView>, Fn, IntView, IntView>);
static_assert(
  !IsImplicitlyConstructible<cuda::std::ranges::zip_transform_view<Fn, IntView, IntView>, Fn, IntView, IntView>);

struct MoveAwareView : cuda::std::ranges::view_base
{
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  TEST_FUNC constexpr MoveAwareView(MoveAwareView&& other)
      : moves(other.moves + 1)
  {
    other.moves = 1;
  }
  TEST_FUNC constexpr MoveAwareView& operator=(MoveAwareView&& other)
  {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  TEST_FUNC constexpr const int* begin() const
  {
    return &moves;
  }
  TEST_FUNC constexpr const int* end() const
  {
    return &moves + 1;
  }
};

template <class View1, class View2, class T, class U>
TEST_FUNC constexpr void constructorTest(T&& buffer1, U&& buffer2)
{
  cuda::std::ranges::zip_transform_view v{MakeTuple{}, View1{buffer1}, View2{buffer2}};
  auto [i, j] = *v.begin();
  assert(i == buffer1[0]);
  assert(j == buffer2[0]);
};

TEST_FUNC constexpr bool test()
{
  int buffer[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[4] = {9, 8, 7, 6};

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer2});
    assert(cuda::std::ranges::equal(
      v, cuda::std::vector{cuda::std::tuple(9), cuda::std::tuple(8), cuda::std::tuple(7), cuda::std::tuple(6)}));
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, cuda::std::views::iota(0));
    assert(cuda::std::ranges::equal(v, cuda::std::vector{1, 2, 3, 4, 5, 6, 7, 8}));
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer2}, cuda::std::ranges::single_view(2.));
    assert(cuda::std::ranges::equal(v, cuda::std::vector{cuda::std::tuple(1, 9, 2.0)}));
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(cuda::std::ranges::empty(v));
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>());
    assert(cuda::std::ranges::empty(v));
  }
  {
    // constructor from views
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SizedRandomAccessView{buffer}, cuda::std::views::iota(0), cuda::std::ranges::single_view(2.));
    auto [i, j, k] = *v.begin();
    assert(i == 1);
    assert(j == 0);
    assert(k == 2.0);
  }

  {
    // arguments are moved once
    MoveAwareView mv;
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, cuda::std::move(mv), MoveAwareView{}};
    auto [numMoves1, numMoves2] = *v.begin();
    assert(numMoves1 == 3); // one move from the local variable to parameter, one move from parameter to member
    assert(numMoves2 == 2);
  }

  // input and forward
  {
    constructorTest<InputCommonView, ForwardSizedView>(buffer, buffer2);
  }

  // bidi and random_access
  {
    constructorTest<BidiCommonView, SizedRandomAccessView>(buffer, buffer2);
  }

  // contiguous
  {
    constructorTest<ContiguousCommonView, ContiguousCommonView>(buffer, buffer2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
