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

// template <class... Types>
// constexpr void swap(const tuple<Types...>& x, const tuple<Types...>& y);

#include <cuda/std/cassert>
#include <cuda/std/tuple>
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

struct Counted
{
  int* calls;
  TEST_FUNC friend constexpr void swap(const Counted& lhs, const Counted& rhs)
  {
    ++*lhs.calls;
    ++*rhs.calls;
  }
};

static_assert(noexcept(swap(cuda::std::declval<const cuda::std::tuple<>&>(),
                            cuda::std::declval<const cuda::std::tuple<>&>())));
static_assert(noexcept(swap(cuda::std::declval<const cuda::std::tuple<int&>&>(),
                            cuda::std::declval<const cuda::std::tuple<int&>&>())));
static_assert(!noexcept(swap(cuda::std::declval<const cuda::std::tuple<MayThrow>&>(),
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
    const cuda::std::tuple<> t0{};
    const cuda::std::tuple<> t1{};
    swap(t0, t1);
  }
  {
    int a = 1;
    int b = 2;
    swap(cuda::std::tie(a), cuda::std::tie(b));
    assert(a == 2);
    assert(b == 1);
  }
  {
    int a  = 1;
    int b  = 2;
    long c = 3;
    long d = 4;
    const cuda::std::tuple<int&, long&> t0{a, c};
    const cuda::std::tuple<int&, long&> t1{b, d};
    swap(t0, t1);
    assert(a == 2);
    assert(b == 1);
    assert(c == 4);
    assert(d == 3);

    // Temporaries model `zip_iterator::operator*`, which returns a prvalue tuple of references.
    swap(cuda::std::tie(a, c), cuda::std::tie(b, d));
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(d == 4);
  }
  return true;
}

TEST_FUNC bool test_const_swappable()
{
  int calls = 0;
  const cuda::std::tuple<Counted, ConstSwappable> t0{Counted{&calls}, ConstSwappable{1}};
  const cuda::std::tuple<Counted, ConstSwappable> t1{Counted{&calls}, ConstSwappable{2}};
  swap(t0, t1);
  assert(calls == 2);
  assert(cuda::std::get<1>(t0).val == 2);
  assert(cuda::std::get<1>(t1).val == 1);
  return true;
}

int main(int, char**)
{
  test_references();
  test_const_swappable();
  static_assert(test_references());
  return 0;
}
