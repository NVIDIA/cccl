//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator*() const noexcept(see below);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

// Test noexcept
// Remarks: Let Is be the pack 0, 1, ..., (sizeof...(Views)-1). The exception specification is equivalent to:
//   noexcept(invoke(*parent_->fun_, *cuda::std::get<Is>(inner_.current_)...)).

template <class ZipTransformView>
_CCCL_CONCEPT DerefNoexcept = _CCCL_REQUIRES_EXPR(
  (ZipTransformView), cuda::std::ranges::iterator_t<ZipTransformView> iter)(requires(noexcept(*iter)));

struct ThrowingDerefIter
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::intptr_t;

  TEST_FUNC int operator*() const noexcept(false);

  TEST_FUNC ThrowingDerefIter& operator++();
  TEST_FUNC void operator++(int);

  TEST_FUNC friend constexpr bool operator==(const ThrowingDerefIter&, const ThrowingDerefIter&) = default;
};

using NoexceptDerefIter = int*;

template <bool NoExceptDeref>
struct TestView : cuda::std::ranges::view_base
{
  using Iter = cuda::std::conditional_t<NoExceptDeref, NoexceptDerefIter, ThrowingDerefIter>;
  TEST_FUNC Iter begin() const;
  TEST_FUNC Iter end() const;
};

template <bool NoExceptCall>
struct TestFn
{
  TEST_FUNC int operator()(auto&&...) const noexcept(NoExceptCall);
};

static_assert(DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<true>, TestView<true>>>);
static_assert(DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<true>, TestView<true>, TestView<true>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<true>, TestView<false>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<false>, TestView<true>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<false>, TestView<false>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<false>, TestView<false>, TestView<true>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<true>, TestView<false>, TestView<true>>>);
static_assert(!DerefNoexcept<cuda::std::ranges::zip_transform_view<TestFn<false>, TestView<false>, TestView<false>>>);

TEST_FUNC constexpr bool test()
{
  cuda::std::array a{1, 2, 3, 4};
  cuda::std::array b{4.1, 3.2, 4.3};
  {
    // Function returns reference
    cuda::std::ranges::zip_transform_view v(GetFirst{}, a);
    auto it                                     = v.begin();
    cuda::std::same_as<int&> decltype(auto) val = *it;
    assert(&val == &a[0]);
  }

  {
    // function returns PRValue
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it                                                              = v.begin();
    cuda::std::same_as<cuda::std::tuple<int, double>> decltype(auto) val = *it;
    assert(val == cuda::std::tuple(1, 4.1));
  }

  {
    // operator* is const
    cuda::std::ranges::zip_transform_view v(GetFirst{}, a);
    const auto it                               = v.begin();
    cuda::std::same_as<int&> decltype(auto) val = *it;
    assert(&val == &a[0]);
  }

  {
    // dereference twice
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it = v.begin();
    assert(*it == cuda::std::tuple(1, 4.1));
    assert(*it == cuda::std::tuple(1, 4.1));
  }

  {
    // back and forth
    cuda::std::ranges::zip_transform_view v(Tie{}, a, b);
    auto it = v.begin();
    assert(&cuda::std::get<0>(*it) == &a[0]);
    assert(&cuda::std::get<1>(*it) == &b[0]);
    ++it;
    assert(&cuda::std::get<0>(*it) == &a[1]);
    assert(&cuda::std::get<1>(*it) == &b[1]);
    --it;
    assert(&cuda::std::get<0>(*it) == &a[0]);
    assert(&cuda::std::get<1>(*it) == &b[0]);
  }

  int buffer[] = {1, 2, 3, 4, 5, 6};
  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1));
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, cuda::std::views::iota(0));
    auto it = v.begin();
    assert(&*it == &buffer[0]);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::single_view(2.));
    auto it = v.begin();
    assert(&cuda::std::get<0>(*it) == &buffer[0]);
    assert(&cuda::std::get<1>(*it) == &buffer[0]);
    assert(cuda::std::get<2>(*it) == 2.0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
