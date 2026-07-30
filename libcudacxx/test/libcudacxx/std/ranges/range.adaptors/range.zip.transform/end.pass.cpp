//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end()
// constexpr auto end() const
//   requires range<const InnerView> &&
//            regular_invocable<const F&, range_reference_t<const Views>...>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "types.h"

template <class T>
_CCCL_CONCEPT HasConstEnd = _CCCL_REQUIRES_EXPR((T), const T& ct)(ct.end());

template <class T>
_CCCL_CONCEPT HasEnd = _CCCL_REQUIRES_EXPR((T), T& t)(t.end());

TEST_FUNC constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // simple test
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::views::iota(0), cuda::std::ranges::single_view(2.));
    assert(v.begin() != v.end());
    assert(cuda::std::as_const(v).begin() != cuda::std::as_const(v).end());
    assert(v.begin() + 1 == v.end());
    assert(cuda::std::as_const(v).begin() + 1 == cuda::std::as_const(v).end());
  }

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(it + 8 == v.end());
    assert(it + 8 == cuda::std::as_const(v).end());
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, cuda::std::views::iota(0));
    auto it = v.begin();
    assert(it + 8 == v.end());
    assert(it + 8 == cuda::std::as_const(v).end());
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::single_view(2.));
    auto it = v.begin();
    assert(it + 1 == v.end());
    assert(it + 1 == cuda::std::as_const(v).end());
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(cuda::std::as_const(v).begin() == cuda::std::as_const(v).end());
  }

  {
    // common_range<InnerView>
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it       = v.begin();
    auto const_it = cuda::std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = cuda::std::as_const(v).end();

    static_assert(!cuda::std::same_as<decltype(it), decltype(const_it)>);
    static_assert(!cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(cuda::std::same_as<decltype(it), decltype(st)>);
    static_assert(cuda::std::same_as<decltype(const_it), decltype(const_st)>);

    assert(it + 8 == st);
    assert(const_it + 8 == const_st);
  }
  {
    // !common_range<InnerView>
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleNonCommon{buffer});
    auto it       = v.begin();
    auto const_it = cuda::std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = cuda::std::as_const(v).end();

    static_assert(!cuda::std::same_as<decltype(it), decltype(const_it)>);
    static_assert(!cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!cuda::std::same_as<decltype(it), decltype(st)>);
    static_assert(!cuda::std::same_as<decltype(const_it), decltype(const_st)>);

    assert(it + 8 == st);
    assert(const_it + 8 == const_st);
  }

  {
    // underlying const R is not a range
    using ZTV = cuda::std::ranges::zip_transform_view<MakeTuple, SimpleCommon, NoConstBeginView>;
    static_assert(HasEnd<ZTV>);
    static_assert(!HasConstEnd<ZTV>);
  }

  {
    // Fn cannot invoke on const range
    using ZTV = cuda::std::ranges::zip_transform_view<NonConstOnlyFn, ConstNonConstDifferentView>;
    static_assert(HasEnd<ZTV>);
    static_assert(!HasConstEnd<ZTV>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
