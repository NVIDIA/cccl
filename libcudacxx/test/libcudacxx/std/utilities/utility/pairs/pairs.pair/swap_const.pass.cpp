//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T1, class T2> struct pair

// constexpr void swap(const pair& p) const;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class _Tp, class = void>
constexpr bool can_const_member_swap = false;

template <class _Tp>
constexpr bool
  can_const_member_swap<_Tp, decltype(cuda::std::declval<const _Tp&>().swap(cuda::std::declval<const _Tp&>()))> = true;

struct ConstSwappable
{
  mutable int val;

  TEST_FUNC friend constexpr void swap(const ConstSwappable& lhs, const ConstSwappable& rhs)
  {
    const int tmp = lhs.val;
    lhs.val       = rhs.val;
    rhs.val       = tmp;
  }
};

struct NonConstSwappable
{
  TEST_FUNC friend void swap(NonConstSwappable&, NonConstSwappable&) {}
};

struct MayThrow
{
  TEST_FUNC friend constexpr void swap(const MayThrow&, const MayThrow&) {}
};

static_assert(noexcept(cuda::std::declval<const cuda::std::pair<int&, int&>&>().swap(
  cuda::std::declval<const cuda::std::pair<int&, int&>&>())));
static_assert(!noexcept(cuda::std::declval<const cuda::std::pair<MayThrow, int&>&>().swap(
  cuda::std::declval<const cuda::std::pair<MayThrow, int&>&>())));

static_assert(cuda::std::is_swappable_v<const cuda::std::pair<int&, int&>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::pair<ConstSwappable, ConstSwappable>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<int, int>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<ConstSwappable, NonConstSwappable>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<NonConstSwappable, ConstSwappable>>);

static_assert(can_const_member_swap<cuda::std::pair<int&, int&>>);
static_assert(can_const_member_swap<cuda::std::pair<ConstSwappable, ConstSwappable>>);
static_assert(!can_const_member_swap<cuda::std::pair<int, int>>);
static_assert(!can_const_member_swap<cuda::std::pair<ConstSwappable, NonConstSwappable>>);
static_assert(!can_const_member_swap<cuda::std::pair<NonConstSwappable, ConstSwappable>>);

TEST_FUNC constexpr bool test_references()
{
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  const cuda::std::pair<int&, int&> p0{a, b};
  const cuda::std::pair<int&, int&> p1{c, d};
  p0.swap(p1);
  assert(a == 3);
  assert(b == 4);
  assert(c == 1);
  assert(d == 2);
  assert(cuda::std::addressof(p0.first) == cuda::std::addressof(a));
  assert(cuda::std::addressof(p1.first) == cuda::std::addressof(c));
  return true;
}

TEST_FUNC bool test_const_swappable()
{
  const cuda::std::pair<ConstSwappable, ConstSwappable> p0{ConstSwappable{1}, ConstSwappable{2}};
  const cuda::std::pair<ConstSwappable, ConstSwappable> p1{ConstSwappable{3}, ConstSwappable{4}};
  p0.swap(p1);
  assert(p0.first.val == 3);
  assert(p0.second.val == 4);
  assert(p1.first.val == 1);
  assert(p1.second.val == 2);
  return true;
}

int main(int, char**)
{
  test_references();
  test_const_swappable();
  static_assert(test_references());
  return 0;
}
