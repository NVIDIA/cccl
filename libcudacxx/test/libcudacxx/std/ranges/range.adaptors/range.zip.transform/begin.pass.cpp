//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin();
// constexpr auto begin() const
//   requires range<const InnerView> &&
//            regular_invocable<const F&, range_reference_t<const Views>...>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "types.h"

template <class T>
_CCCL_CONCEPT HasConstBegin = _CCCL_REQUIRES_EXPR((T), const T& ct)(ct.begin());

template <class T>
_CCCL_CONCEPT HasBegin = _CCCL_REQUIRES_EXPR((T), T& t)(t.begin());

TEST_FUNC constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // all underlying iterators should be at the begin position
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer}, cuda::std::views::iota(0), cuda::std::ranges::single_view(2.));
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1, 0, 2.0));

    auto const_it = cuda::std::as_const(v).begin();
    assert(*const_it == *it);

    static_assert(!cuda::std::same_as<decltype(it), decltype(const_it)>);
  }

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1));
    auto cit = cuda::std::as_const(v).begin();
    assert(*cit == cuda::std::make_tuple(1));
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, cuda::std::views::iota(0));
    auto it = v.begin();
    assert(&*it == &buffer[0]);
    auto cit = cuda::std::as_const(v).begin();
    assert(&*cit == &buffer[0]);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, cuda::std::ranges::single_view(2.));
    auto it = v.begin();
    assert(&cuda::std::get<0>(*it) == &buffer[0]);
    assert(&cuda::std::get<1>(*it) == &buffer[0]);
    assert(cuda::std::get<2>(*it) == 2.0);
    auto cit = cuda::std::as_const(v).begin();
    assert(&cuda::std::get<0>(*cit) == &buffer[0]);
    assert(&cuda::std::get<1>(*cit) == &buffer[0]);
    assert(cuda::std::get<2>(*cit) == 2.0);
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
    // underlying const R is not a range
    using ZTV = cuda::std::ranges::zip_transform_view<MakeTuple, SimpleCommon, NoConstBeginView>;
    static_assert(HasBegin<ZTV>);
    static_assert(!HasConstBegin<ZTV>);
  }

  {
    // Fn cannot be invoked on const range
    using ZTV = cuda::std::ranges::zip_transform_view<NonConstOnlyFn, ConstNonConstDifferentView>;
    static_assert(HasBegin<ZTV>);
    static_assert(!HasConstBegin<ZTV>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
