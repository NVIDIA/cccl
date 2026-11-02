//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T1, class T2>
// constexpr void swap(const pair<T1, T2>& x, const pair<T1, T2>& y);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

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

static_assert(noexcept(swap(cuda::std::declval<const cuda::std::pair<int&, int&>&>(),
                            cuda::std::declval<const cuda::std::pair<int&, int&>&>())));
static_assert(!noexcept(swap(cuda::std::declval<const cuda::std::pair<MayThrow, int&>&>(),
                             cuda::std::declval<const cuda::std::pair<MayThrow, int&>&>())));

static_assert(cuda::std::is_swappable_v<const cuda::std::pair<int&, int&>>);
static_assert(cuda::std::is_swappable_v<const cuda::std::pair<ConstSwappable, ConstSwappable>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<int, int>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<ConstSwappable, NonConstSwappable>>);
static_assert(!cuda::std::is_swappable_v<const cuda::std::pair<NonConstSwappable, ConstSwappable>>);

TEST_FUNC constexpr bool test_references()
{
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  swap(cuda::std::pair<int&, int&>{a, b}, cuda::std::pair<int&, int&>{c, d});
  assert(a == 3);
  assert(b == 4);
  assert(c == 1);
  assert(d == 2);

  const cuda::std::pair<int&, int&> p0{a, b};
  const cuda::std::pair<int&, int&> p1{c, d};
  cuda::std::swap(p0, p1);
  assert(a == 1);
  assert(b == 2);
  assert(c == 3);
  assert(d == 4);
  return true;
}

TEST_FUNC bool test_const_swappable()
{
  const cuda::std::pair<ConstSwappable, ConstSwappable> p0{ConstSwappable{1}, ConstSwappable{2}};
  const cuda::std::pair<ConstSwappable, ConstSwappable> p1{ConstSwappable{3}, ConstSwappable{4}};
  using cuda::std::swap;
  swap(p0, p1);
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
