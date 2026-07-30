//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types>
// template <class... UTypes>
//   constexpr explicit(see below) tuple<Types>::tuple(const
//   tuple<UTypes...>&&);
//
// Constraints:
//  sizeof...(Types) equals sizeof...(UTypes) &&
//  (is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...) is true &&
//  (
//    sizeof...(Types) is not 1 ||
//    (
//      !is_convertible_v<decltype(u), T> &&
//      !is_constructible_v<T, decltype(u)> &&
//      !is_same_v<T, U>
//    )
//  )

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "copy_move_types.h"
#include "test_macros.h"

// test: The expression inside explicit is equivalent to:
// !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
static_assert(
  cuda::std::is_convertible_v<const cuda::std::tuple<ConstMove>&&, cuda::std::tuple<ConvertibleFrom<ConstMove>>>);

static_assert(cuda::std::is_convertible_v<const cuda::std::tuple<ConstMove, ConstMove>&&,
                                          cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>>);

static_assert(!cuda::std::is_convertible_v<const cuda::std::tuple<MutableCopy>&&,
                                           cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>>>);

static_assert(
  !cuda::std::is_convertible_v<const cuda::std::tuple<ConstMove, ConstMove>&&,
                               cuda::std::tuple<ConvertibleFrom<ConstMove>, ExplicitConstructibleFrom<ConstMove>>>);

TEST_FUNC constexpr bool test()
{
  // test implicit conversions.
  // sizeof...(Types) == 1
  {
    const cuda::std::tuple<ConstMove> t1{1};
    cuda::std::tuple<ConvertibleFrom<ConstMove>> t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  // test implicit conversions.
  // sizeof...(Types) > 1
  {
    const cuda::std::tuple<ConstMove, int> t1{1, 2};
    cuda::std::tuple<ConvertibleFrom<ConstMove>, int> t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2) == 2);
  }

  // test explicit conversions.
  // sizeof...(Types) == 1
  {
    const cuda::std::tuple<ConstMove> t1{1};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>> t2{cuda::std::move(t1)};
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  // test explicit conversions.
  // sizeof...(Types) > 1
  {
    const cuda::std::tuple<ConstMove, int> t1{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, int> t2{cuda::std::move(t1)};
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2) == 2);
  }

  // test constraints

  // sizeof...(Types) != sizeof...(UTypes)
  static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, int>, const cuda::std::tuple<int>&&>);
  static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, int, int>, const cuda::std::tuple<int, int>&&>);

  // !(is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...)
  static_assert(
    !cuda::std::is_constructible_v<cuda::std::tuple<int, NoConstructorFromInt>, const cuda::std::tuple<int, int>&&>);

  // sizeof...(Types) == 1 && other branch of "||" satisfied
  {
    const cuda::std::tuple<TracedCopyMove> t1{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>> t2{cuda::std::move(t1)};
    assert(constMoveCtrCalled(cuda::std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_same_v<T, U>
  {
    const cuda::std::tuple<TracedCopyMove> t1{};
    cuda::std::tuple<TracedCopyMove> t2{t1};
    assert(!constMoveCtrCalled(cuda::std::get<0>(t2)));
  }

  // sizeof...(Types) != 1
  {
    const cuda::std::tuple<TracedCopyMove, TracedCopyMove> t1{};
    cuda::std::tuple<TracedCopyMove, TracedCopyMove> t2{cuda::std::move(t1)};
    assert(constMoveCtrCalled(cuda::std::get<0>(t2)));
  }

  // This segfaults MSVC trying to determine is_nothrow_constructible
#if !TEST_COMPILER(MSVC)
  // sizeof...(Types) == 1 && is_convertible_v<decltype(u), T>
  {
    const cuda::std::tuple<CvtFromConstTupleRefRef> t1{};
    cuda::std::tuple<ConvertibleFrom<CvtFromConstTupleRefRef>> t2{cuda::std::move(t1)};
    assert(!constMoveCtrCalled(cuda::std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_constructible_v<decltype(u), T>
  {
    const cuda::std::tuple<ExplicitCtrFromConstTupleRefRef> t1{};
    cuda::std::tuple<ConvertibleFrom<ExplicitCtrFromConstTupleRefRef>> t2{cuda::std::move(t1)};
    assert(!constMoveCtrCalled(cuda::std::get<0>(t2).v));
  }
#endif // !TEST_COMPILER(MSVC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
