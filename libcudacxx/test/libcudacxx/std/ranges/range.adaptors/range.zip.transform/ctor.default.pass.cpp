//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// zip_transform_view() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "types.h"

_CCCL_GLOBAL_CONSTANT int buff[] = {1, 2, 3};

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

struct NonDefaultConstructibleView : cuda::std::ranges::view_base
{
  NonDefaultConstructibleView() = delete;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct DefaultConstructibleFn
{
  template <class... T>
  TEST_FUNC constexpr int operator()(const T&... x) const
  {
    return (x + ...);
  }
};

struct NonDefaultConstructibleFn
{
  NonDefaultConstructibleFn() = delete;
  template <class... T>
  TEST_FUNC constexpr int operator()(const T&... x) const;
};

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the zip_view's constructor.
static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::zip_transform_view< //
                DefaultConstructibleFn, //
                DefaultConstructibleView>>);
static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::zip_transform_view< //
                DefaultConstructibleFn, //
                DefaultConstructibleView,
                DefaultConstructibleView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_transform_view< //
                NonDefaultConstructibleFn, //
                DefaultConstructibleView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_transform_view< //
                DefaultConstructibleFn, //
                NonDefaultConstructibleView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_transform_view< //
                DefaultConstructibleFn, //
                DefaultConstructibleView,
                NonDefaultConstructibleView>>);

TEST_FUNC constexpr bool test()
{
  {
    using View =
      cuda::std::ranges::zip_transform_view<DefaultConstructibleFn, DefaultConstructibleView, DefaultConstructibleView>;
    View v = View(); // the default constructor is not explicit
    assert(v.size() == 3);
    auto it = v.begin();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it == 6);
  }

  {
    // one range
    using View = cuda::std::ranges::zip_transform_view<MakeTuple, DefaultConstructibleView>;
    View v     = View(); // the default constructor is not explicit
    auto it    = v.begin();
    assert(*it == cuda::std::make_tuple(1));
  }

  {
    // two ranges
    using View =
      cuda::std::ranges::zip_transform_view<MakeTuple, DefaultConstructibleView, cuda::std::ranges::iota_view<int>>;
    View v  = View(); // the default constructor is not explicit
    auto it = v.begin();
    assert(*it == cuda::std::tuple(1, 0));
  }

  {
    // three ranges
    using View = cuda::std::ranges::zip_transform_view<MakeTuple,
                                                       DefaultConstructibleView,
                                                       DefaultConstructibleView,
                                                       cuda::std::ranges::iota_view<int>>;
    View v     = View(); // the default constructor is not explicit
    auto it    = v.begin();
    assert(*it == cuda::std::tuple(1, 1, 0));
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range at the beginning
    using View = cuda::std::ranges::zip_transform_view<MakeTuple,
                                                       cuda::std::ranges::empty_view<int>,
                                                       DefaultConstructibleView,
                                                       DefaultConstructibleView>;
    View v     = View(); // the default constructor is not explicit
    assert(v.empty());
  }

  {
    // empty range in the middle
    using View =
      cuda::std::ranges::zip_transform_view<MakeTuple,
                                            DefaultConstructibleView,
                                            cuda::std::ranges::empty_view<int>,
                                            DefaultConstructibleView,
                                            DefaultConstructibleView>;
    View v = View(); // the default constructor is not explicit
    assert(v.empty());
  }

  {
    // empty range at the end
    using View = cuda::std::ranges::zip_transform_view<MakeTuple,
                                                       DefaultConstructibleView,
                                                       DefaultConstructibleView,
                                                       cuda::std::ranges::empty_view<int>>;
    View v     = View(); // the default constructor is not explicit
    assert(v.empty());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
