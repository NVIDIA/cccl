//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// constexpr void swap(const tuple& rhs) const;

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

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

static_assert(
  noexcept(cuda::std::declval<const cuda::std::tuple<>&>().swap(cuda::std::declval<const cuda::std::tuple<>&>())));
static_assert(noexcept(
  cuda::std::declval<const cuda::std::tuple<int&>&>().swap(cuda::std::declval<const cuda::std::tuple<int&>&>())));
static_assert(!noexcept(cuda::std::declval<const cuda::std::tuple<MayThrow>&>().swap(
  cuda::std::declval<const cuda::std::tuple<MayThrow>&>())));

static_assert(cuda::std::is_swappable_v<const cuda::std::tuple<>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::tuple<int&>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::tuple<int&, long&>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::tuple<ConstSwappable>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::tuple<ConstSwappable, int&>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::tuple<int>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::tuple<NonConstSwappable>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::tuple<ConstSwappable, NonConstSwappable>>);

TEST_FUNC constexpr bool test_references()
{
  {
    cuda::std::tuple<> t0{};
    const cuda::std::tuple<> t1{};
    t1.swap(t0);
  }
  {
    int a = 1;
    int b = 2;
    const cuda::std::tuple<int&> t0{a};
    const cuda::std::tuple<int&> t1{b};
    t0.swap(t1);
    assert(a == 2);
    assert(b == 1);
    assert(cuda::std::addressof(cuda::std::get<0>(t0)) == cuda::std::addressof(a));
    assert(cuda::std::addressof(cuda::std::get<0>(t1)) == cuda::std::addressof(b));
  }
  {
    int a  = 1;
    int b  = 2;
    long c = 3;
    long d = 4;
    const cuda::std::tuple<int&, long&> t0{a, c};
    const cuda::std::tuple<int&, long&> t1{b, d};
    t0.swap(t1);
    assert(a == 2);
    assert(b == 1);
    assert(c == 4);
    assert(d == 3);
  }
  {
    int a  = 1;
    int b  = 2;
    long c = 3;
    long d = 4;
    int e  = 5;
    int f  = 6;
    const cuda::std::tuple<int&, long&, int&> t0{a, c, e};
    const cuda::std::tuple<int&, long&, int&> t1{b, d, f};
    t0.swap(t1);
    assert(a == 2);
    assert(c == 4);
    assert(e == 6);
    assert(b == 1);
    assert(d == 3);
    assert(f == 5);
  }
  return true;
}

TEST_FUNC bool test_const_swappable()
{
  {
    const cuda::std::tuple<ConstSwappable> t0{ConstSwappable{1}};
    const cuda::std::tuple<ConstSwappable> t1{ConstSwappable{2}};
    t0.swap(t1);
    assert(cuda::std::get<0>(t0).val == 2);
    assert(cuda::std::get<0>(t1).val == 1);
  }
  {
    int a = 1;
    int b = 2;
    const cuda::std::tuple<ConstSwappable, int&> t0{ConstSwappable{3}, a};
    const cuda::std::tuple<ConstSwappable, int&> t1{ConstSwappable{4}, b};
    t0.swap(t1);
    assert(cuda::std::get<0>(t0).val == 4);
    assert(cuda::std::get<0>(t1).val == 3);
    assert(a == 2);
    assert(b == 1);
  }
  return true;
}

int main(int, char**)
{
  test_references();
  test_const_swappable();
  static_assert(test_references());
  return 0;
}
