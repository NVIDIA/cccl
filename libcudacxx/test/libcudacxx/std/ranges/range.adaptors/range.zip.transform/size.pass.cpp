//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size() requires sized_range<InnerView>
// constexpr auto size() const requires sized_range<const InnerView>

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "types.h"

_CCCL_GLOBAL_CONSTANT int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

struct SizedView : cuda::std::ranges::view_base
{
  cuda::std::size_t size_ = 0;
  TEST_FUNC constexpr SizedView(cuda::std::size_t s)
      : size_(s)
  {}
  TEST_FUNC constexpr auto begin() const
  {
    return buffer;
  }
  TEST_FUNC constexpr auto end() const
  {
    return buffer + size_;
  }
};

struct SizedNonConst : cuda::std::ranges::view_base
{
  using iterator          = forward_iterator<int*>;
  cuda::std::size_t size_ = 0;
  TEST_FUNC constexpr SizedNonConst(cuda::std::size_t s)
      : size_(s)
  {}
  TEST_FUNC constexpr auto begin() const
  {
    return iterator{buffer};
  }
  TEST_FUNC constexpr auto end() const
  {
    return iterator{buffer + size_};
  }
  TEST_FUNC constexpr cuda::std::size_t size()
  {
    return size_;
  }
};

struct ConstNonConstDifferentSize : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr auto begin() const
  {
    return buffer;
  }
  TEST_FUNC constexpr auto end() const
  {
    return buffer + 8;
  }

  TEST_FUNC constexpr auto size()
  {
    return 5;
  }
  TEST_FUNC constexpr auto size() const
  {
    return 6;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    assert(v.size() == 9);
    assert(cuda::std::as_const(v).size() == 9);
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, SizedView(3));
    assert(v.size() == 3);
    assert(cuda::std::as_const(v).size() == 3);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer}, SizedView{6}, cuda::std::ranges::single_view(2.));
    assert(v.size() == 1);
    assert(cuda::std::as_const(v).size() == 1);
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    assert(v.size() == 0);
    assert(cuda::std::as_const(v).size() == 0);
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(v.size() == 0);
    assert(cuda::std::as_const(v).size() == 0);
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(v.size() == 0);
    assert(cuda::std::as_const(v).size() == 0);
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>());
    assert(v.size() == 0);
    assert(cuda::std::as_const(v).size() == 0);
  }

  {
    // const-view non-sized range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SizedNonConst(2), SizedView(3));
    assert(v.size() == 2);
    static_assert(cuda::std::ranges::sized_range<decltype(v)>);
    static_assert(!cuda::std::ranges::sized_range<decltype(cuda::std::as_const(v))>);
  }

  {
    // const/non-const has different sizes
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, ConstNonConstDifferentSize{});
    assert(v.size() == 5);
    assert(cuda::std::as_const(v).size() == 6);
  }

  {
    // underlying range not sized
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, InputCommonView{buffer});
    static_assert(!cuda::std::ranges::sized_range<decltype(v)>);
    static_assert(!cuda::std::ranges::sized_range<decltype(cuda::std::as_const(v))>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
