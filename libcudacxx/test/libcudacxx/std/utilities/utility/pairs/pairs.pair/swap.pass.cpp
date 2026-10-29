//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// constexpr void swap(pair& p);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct NotSwappable
{
  NotSwappable(const NotSwappable&)            = delete;
  NotSwappable& operator=(const NotSwappable&) = delete;
};

template <class _Tp, class = void>
constexpr bool can_member_swap = false;

template <class _Tp>
constexpr bool can_member_swap<_Tp, decltype(cuda::std::declval<_Tp&>().swap(cuda::std::declval<_Tp&>()))> = true;

struct S
{
  int i;
  TEST_FUNC constexpr S()
      : i(0)
  {}
  TEST_FUNC constexpr S(int j)
      : i(j)
  {}
  TEST_FUNC constexpr bool operator==(int x) const
  {
    return i == x;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    using P1 = cuda::std::pair<int, short>;
    P1 p1(3, static_cast<short>(4));
    P1 p2(5, static_cast<short>(6));
    p1.swap(p2);
    assert(p1.first == 5);
    assert(p1.second == 6);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }

  {
    using P1 = cuda::std::pair<int, S>;
    P1 p1(3, S(4));
    P1 p2(5, S(6));
    p1.swap(p2);
    assert(p1.first == 5);
    assert(p1.second == 6);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }

  return true;
}

static_assert(cuda::std::is_swappable_v<cuda::std::pair<int, int>>);
static_assert(can_member_swap<cuda::std::pair<int, int>>);
static_assert(
  noexcept(cuda::std::declval<cuda::std::pair<int, int>&>().swap(cuda::std::declval<cuda::std::pair<int, int>&>())));
static_assert(!cuda::std::is_swappable_v<cuda::std::pair<NotSwappable, int>>);
static_assert(!can_member_swap<cuda::std::pair<NotSwappable, int>>);

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
