//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr iterator& operator+=(difference_type x) requires random_access_range<Base>;
//  constexpr iterator& operator-=(difference_type x) requires random_access_range<Base>;
//  friend constexpr iterator operator+(const iterator& i, difference_type n)
//    requires random_access_range<Base>;
//  friend constexpr iterator operator+(difference_type n, const iterator& i)
//    requires random_access_range<Base>;
//  friend constexpr iterator operator-(const iterator& i, difference_type n)
//    requires random_access_range<Base>;
//  friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//    requires sized_sentinel_for<ziperator<Const>, ziperator<Const>>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "../types.h"

template <class T, class U>
_CCCL_CONCEPT canPlusEqual = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)((t += u));

template <class T, class U>
_CCCL_CONCEPT canPlus = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)((t + u));

template <class T, class U>
_CCCL_CONCEPT canMinusEqual = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)((t -= u));

template <class T, class U>
_CCCL_CONCEPT canMinus = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)((t - u));

TEST_FUNC constexpr bool test()
{
  int buffer1[5] = {1, 2, 3, 4, 5};
  SizedRandomAccessView a{buffer1};
  static_assert(cuda::std::ranges::random_access_range<decltype(a)>);

  cuda::std::array b{4.1, 3.2, 4.3, 0.1, 0.2};
  static_assert(cuda::std::ranges::contiguous_range<decltype(b)>);

  {
    // operator+(x, n) and operator+=
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it1   = v.begin();
    using Iter = decltype(it1);

    cuda::std::same_as<Iter> decltype(auto) it2 = it1 + 3;
    assert(*it2 == cuda::std::tuple(4, 0.1));

    cuda::std::same_as<Iter> decltype(auto) it3 = 3 + it1;
    assert(*it3 == cuda::std::tuple(4, 0.1));

    cuda::std::same_as<Iter&> decltype(auto) it1_ref = it1 += 3;
    assert(&it1_ref == &it1);
    assert(*it1_ref == cuda::std::tuple(4, 0.1));
    assert(*it1 == cuda::std::tuple(4, 0.1));

    static_assert(canPlus<Iter, cuda::std::intptr_t>);
    static_assert(canPlusEqual<Iter, cuda::std::intptr_t>);
  }

  {
    // operator-(x, n) and operator-=
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it1   = v.end();
    using Iter = decltype(it1);

    cuda::std::same_as<Iter> decltype(auto) it2 = it1 - 3;
    assert(*it2 == cuda::std::tuple(3, 4.3));

    cuda::std::same_as<Iter&> decltype(auto) it1_ref = it1 -= 3;
    assert(&it1_ref == &it1);
    assert(*it1_ref == cuda::std::tuple(3, 4.3));
    assert(*it1 == cuda::std::tuple(3, 4.3));

    static_assert(canMinusEqual<Iter, cuda::std::intptr_t>);
    static_assert(canMinus<Iter, cuda::std::intptr_t>);
  }

  {
    // operator-(x, y)
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    assert((v.end() - v.begin()) == 5);

    auto it1 = v.begin() + 2;
    auto it2 = v.end() - 1;

    using Iter = decltype(it1);

    cuda::std::same_as<cuda::std::iter_difference_t<Iter>> decltype(auto) n = it1 - it2;
    assert(n == -2);
  }

  {
    // one range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer1});
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1));

    it += 4;
    assert(*it == cuda::std::make_tuple(5));

    it -= 1;
    assert(*it == cuda::std::make_tuple(4));

    auto it2 = it - 2;
    assert(*it2 == cuda::std::make_tuple(2));

    auto it3 = 3 + it2;
    assert(*it3 == cuda::std::make_tuple(5));

    assert(it3 - it2 == 3);
  }

  {
    // two ranges
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer1}, cuda::std::views::iota(0));
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1, 0));

    it += 4;
    assert(*it == cuda::std::make_tuple(5, 4));

    it -= 1;
    assert(*it == cuda::std::make_tuple(4, 3));

    auto it2 = it - 2;
    assert(*it2 == cuda::std::make_tuple(2, 1));

    auto it3 = 3 + it2;
    assert(*it3 == cuda::std::make_tuple(5, 4));

    assert(it3 - it2 == 3);
  }

  {
    // three ranges
    cuda::std::ranges::zip_transform_view v(
      Tie{}, SimpleCommon{buffer1}, SimpleCommon{buffer1}, cuda::std::ranges::single_view(2.));
    auto it = v.begin();
    assert(*it == cuda::std::make_tuple(1, 1, 2.0));

    it += 1;
    assert(it == v.end());

    it -= 1;
    assert(it == v.begin());

    auto it2 = it + 1;
    assert(it2 == v.end());

    auto it3 = it2 - 1;
    assert(it3 == v.begin());

    assert(it3 - it2 == -1);
  }

  {
    // single empty range
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, cuda::std::ranges::empty_view<int>());
    auto it  = v.begin();
    auto it2 = v.end();
    assert(it2 - it == 0);
  }

  {
    // empty range at the beginning
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer1}, SimpleCommon{buffer1});
    auto it  = v.begin();
    auto it2 = v.end();
    assert(it2 - it == 0);
  }

  {
    // empty range in the middle
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer1}, cuda::std::ranges::empty_view<int>(), SimpleCommon{buffer1});
    auto it  = v.begin();
    auto it2 = v.end();
    assert(it2 - it == 0);
  }

  {
    // empty range at the end
    cuda::std::ranges::zip_transform_view v(
      MakeTuple{}, SimpleCommon{buffer1}, SimpleCommon{buffer1}, cuda::std::ranges::empty_view<int>());
    auto it  = v.begin();
    auto it2 = v.end();
    assert(it2 - it == 0);
  }
  {
    // One of the ranges is not random access
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b, ForwardSizedView{buffer1});
    auto it1   = v.begin();
    using Iter = decltype(it1);
    static_assert(!canPlus<Iter, cuda::std::intptr_t>);
    static_assert(!canPlus<cuda::std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, cuda::std::intptr_t>);
    static_assert(!canMinus<Iter, cuda::std::intptr_t>);
    static_assert(canMinus<Iter, Iter>);
    static_assert(!canMinusEqual<Iter, cuda::std::intptr_t>);

    auto it2 = ++v.begin();
    assert((it2 - it1) == 1);
  }

  {
    // One of the ranges does not have sized sentinel
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, a, b, InputCommonView{buffer1});
    using Iter = decltype(v.begin());
    static_assert(!canPlus<Iter, cuda::std::intptr_t>);
    static_assert(!canPlus<cuda::std::intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, cuda::std::intptr_t>);
    static_assert(!canMinus<Iter, cuda::std::intptr_t>);
    static_assert(!canMinus<Iter, Iter>);
    static_assert(!canMinusEqual<Iter, cuda::std::intptr_t>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
