//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types>
// template <class U1, class U2>
// constexpr explicit(see below) tuple<Types...>::tuple(pair<U1, U2>& u);

// Constraints:
// - sizeof...(Types) is 2 and
// - is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// - is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "copy_move_types.h"
#include "test_macros.h"

// test constraints
// sizeof...(Types) == 2
static_assert(cuda::std::is_constructible_v<cuda::std::tuple<MutableCopy, int>, cuda::std::pair<MutableCopy, int>&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<MutableCopy>, cuda::std::pair<MutableCopy, int>&>);

static_assert(
  !cuda::std::is_constructible_v<cuda::std::tuple<MutableCopy, int, int>, cuda::std::pair<MutableCopy, int>&>);

// test constraints
// is_constructible_v<T0, decltype(get<0>(FWD(u)))> is true and
// is_constructible_v<T1, decltype(get<1>(FWD(u)))> is true.
static_assert(cuda::std::is_constructible_v<cuda::std::tuple<int, int>, cuda::std::pair<int, int>&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<NoConstructorFromInt, int>, cuda::std::pair<int, int>&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, NoConstructorFromInt>, cuda::std::pair<int, int>&>);

static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<NoConstructorFromInt, NoConstructorFromInt>,
                                             cuda::std::pair<int, int>&>);

// test: The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(u))), T0> ||
// !is_convertible_v<decltype(get<1>(FWD(u))), T1>
static_assert(cuda::std::is_convertible_v<cuda::std::pair<MutableCopy, MutableCopy>&,
                                          cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(
  !cuda::std::is_convertible_v<cuda::std::pair<MutableCopy, MutableCopy>&,
                               cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(
  !cuda::std::is_convertible_v<cuda::std::pair<MutableCopy, MutableCopy>&,
                               cuda::std::tuple<ConvertibleFrom<MutableCopy>, ExplicitConstructibleFrom<MutableCopy>>>);

TEST_FUNC constexpr bool test()
{
  // test implicit conversions.
  {
    cuda::std::pair<MutableCopy, int> p{1, 2};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> t = p;
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

  // test explicit conversions.
  {
    cuda::std::pair<MutableCopy, int> p{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> t{p};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

  // non const overload should be called
  {
    cuda::std::pair<TracedCopyMove, TracedCopyMove> p;
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = p;
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(t).v));
    assert(nonConstCopyCtrCalled(cuda::std::get<1>(t)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
