//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires equality_comparable<iterator_t<Base>>;
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires random_access_range<Base> && three_way_comparable<iterator_t<Base>>;

#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2>
__host__ __device__ constexpr void compareOperatorTest(const Iter1& iter1, const Iter2& iter2)
{
  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));

  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));

  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);

  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);

  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);

  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));
}

template <class Iter1, class Iter2>
__host__ __device__ constexpr void inequalityOperatorsDoNotExistTest(const Iter1& iter1, const Iter2& iter2)
{
  static_assert(!cuda::std::is_invocable_v<cuda::std::less<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::less_equal<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater_equal<>, Iter1, Iter2>);
}

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> ts[] = {{1}, {2}, {3}};

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It       = three_way_contiguous_iterator<cuda::std::tuple<int>*>;
    using Subrange = cuda::std::ranges::subrange<It>;
    static_assert(cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::elements_view<Subrange, 0>;
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);

    auto ev    = Subrange{It{&ts[0]}, It{&ts[0] + 3}} | cuda::std::views::elements<0>;
    auto iter1 = ev.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

  {
    // Test an old-school iterator with no operator<=>; the elements view iterator shouldn't have
    // operator<=> either.
    using It       = random_access_iterator<cuda::std::tuple<int>*>;
    using Subrange = cuda::std::ranges::subrange<It>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    static_assert(!cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::elements_view<Subrange, 0>;
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

    auto ev    = Subrange{It{ts}, It{ts + 3}} | cuda::std::views::elements<0>;
    auto iter1 = ev.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // non random_access_range
    using It       = bidirectional_iterator<cuda::std::tuple<int>*>;
    using Subrange = cuda::std::ranges::subrange<It>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    static_assert(!cuda::std::ranges::random_access_range<Subrange>);
    using R = cuda::std::ranges::elements_view<Subrange, 0>;
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

    auto ev = Subrange{It{ts}, It{ts + 1}} | cuda::std::views::elements<0>;

    auto it1 = ev.begin();
    auto it2 = ev.end();

    assert(it1 == it1);
    assert(!(it1 != it1));
    assert(it2 == it2);
    assert(!(it2 != it2));

    assert(it1 != it2);

    ++it1;
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // underlying iterator does not support ==
    using It       = cpp20_input_iterator<cuda::std::tuple<int>*>;
    using Sent     = sentinel_wrapper<It>;
    using Subrange = cuda::std::ranges::subrange<It, Sent>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    using R = cuda::std::ranges::elements_view<Subrange, 0>;
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !TEST_HAS_NO_SPACESHIP_OPERATOR

    auto ev = Subrange{It{ts}, Sent{It{ts + 1}}} | cuda::std::views::elements<0>;
    auto it = ev.begin();

    using ElemIter = decltype(it);
    static_assert(!cuda::std::invocable<cuda::std::equal_to<>, ElemIter, ElemIter>);
    static_assert(!cuda::std::invocable<cuda::std::not_equal_to<>, ElemIter, ElemIter>);
    inequalityOperatorsDoNotExistTest(it, it);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
