//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types>
// template <class... UTypes>
//   constexpr explicit(see below) tuple<Types>::tuple(tuple<UTypes...>&);
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
#include <cuda/std/type_traits>

#include "copy_move_types.h"
#include "test_macros.h"

struct ImplicitLvalue
{
  int* ptr_{};

  TEST_FUNC constexpr ImplicitLvalue(int& x)
      : ptr_(&x)
  {}

  TEST_FUNC ImplicitLvalue(const int&) = delete;
};

struct ExplicitLvalue
{
  int* ptr_{};

  TEST_FUNC explicit constexpr ExplicitLvalue(int& x)
      : ptr_(&x)
  {}

  TEST_FUNC ExplicitLvalue(const int&) = delete;
};

// test: The expression inside explicit is equivalent to:
// !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
static_assert(
  cuda::std::is_convertible_v<cuda::std::tuple<MutableCopy>&, cuda::std::tuple<ConvertibleFrom<MutableCopy>>>);

static_assert(cuda::std::is_convertible_v<cuda::std::tuple<MutableCopy, MutableCopy>&,
                                          cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(!cuda::std::is_convertible_v<cuda::std::tuple<MutableCopy>&,
                                           cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>>>);

static_assert(
  !cuda::std::is_convertible_v<cuda::std::tuple<MutableCopy, MutableCopy>&,
                               cuda::std::tuple<ConvertibleFrom<MutableCopy>, ExplicitConstructibleFrom<MutableCopy>>>);

TEST_FUNC constexpr bool test()
{
  {
    using Source = cuda::std::tuple<int>;
    using Dest   = cuda::std::tuple<int&>;

    static_assert(cuda::std::is_constructible_v<Dest, Source&>);
    static_assert(cuda::std::is_convertible_v<Source&, Dest>);
    static_assert(!cuda::std::is_constructible_v<Dest, const Source&>);

    Source src(42);
    Dest dst = src;
    assert(&cuda::std::get<0>(dst) == &cuda::std::get<0>(src));

    cuda::std::get<0>(src) = 43;
    assert(cuda::std::get<0>(dst) == 43);
  }
  {
    using Source = cuda::std::tuple<int>;
    using Dest   = cuda::std::tuple<ImplicitLvalue>;

    static_assert(cuda::std::is_constructible_v<Dest, Source&>);
    static_assert(cuda::std::is_convertible_v<Source&, Dest>);
    static_assert(!cuda::std::is_constructible_v<Dest, const Source&>);

    Source src(42);
    Dest dst = src;
    assert(cuda::std::get<0>(dst).ptr_ == &cuda::std::get<0>(src));
  }
  {
    using Source = cuda::std::tuple<int>;
    using Dest   = cuda::std::tuple<ExplicitLvalue>;

    static_assert(cuda::std::is_constructible_v<Dest, Source&>);
    static_assert(!cuda::std::is_convertible_v<Source&, Dest>);
    static_assert(!cuda::std::is_constructible_v<Dest, const Source&>);

    Source src(42);
    Dest dst(src);
    assert(cuda::std::get<0>(dst).ptr_ == &cuda::std::get<0>(src));
  }

  // test implicit conversions.
  // sizeof...(Types) == 1
  {
    cuda::std::tuple<MutableCopy> t1{1};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>> t2 = t1;
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  // test implicit conversions.
  // sizeof...(Types) > 1
  {
    cuda::std::tuple<MutableCopy, int> t1{1, 2};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>, int> t2 = t1;
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2) == 2);
  }

  // test explicit conversions.
  // sizeof...(Types) == 1
  {
    cuda::std::tuple<MutableCopy> t1{1};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>> t2{t1};
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  // test explicit conversions.
  // sizeof...(Types) > 1
  {
    cuda::std::tuple<MutableCopy, int> t1{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, int> t2{t1};
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2) == 2);
  }

  // test constraints

  // sizeof...(Types) != sizeof...(UTypes)
  static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, int>, cuda::std::tuple<int>&>);
  static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int>, cuda::std::tuple<int, int>&>);
  static_assert(!cuda::std::is_constructible_v<cuda::std::tuple<int, int, int>, cuda::std::tuple<int, int>&>);

  // !(is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...)
  static_assert(
    !cuda::std::is_constructible_v<cuda::std::tuple<int, NoConstructorFromInt>, cuda::std::tuple<int, int>&>);

  // sizeof...(Types) == 1 && other branch of "||" satisfied
  {
    cuda::std::tuple<TracedCopyMove> t1{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>> t2{t1};
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_same_v<T, U>
  {
    cuda::std::tuple<TracedCopyMove> t1{};
    cuda::std::tuple<TracedCopyMove> t2{t1};
    assert(!nonConstCopyCtrCalled(cuda::std::get<0>(t2)));
  }

  // sizeof...(Types) != 1
  {
    cuda::std::tuple<TracedCopyMove, TracedCopyMove> t1{};
    cuda::std::tuple<TracedCopyMove, TracedCopyMove> t2{t1};
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(t2)));
  }

  // These two test points cause gcc to ICE, because it cannot follow the (intentionally)
  // pathological construction chains and ends up thinking that __tuple_constructible is
  // self-referential.
#if !TEST_COMPILER(GCC, <, 13)
  // sizeof...(Types) == 1 && is_convertible_v<decltype(u), T>
  {
    cuda::std::tuple<CvtFromTupleRef> t1{};
    cuda::std::tuple<ConvertibleFrom<CvtFromTupleRef>> t2{t1};
    assert(!nonConstCopyCtrCalled(cuda::std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_constructible_v<decltype(u), T>
  {
    cuda::std::tuple<ExplicitCtrFromTupleRef> t1{};
    cuda::std::tuple<ConvertibleFrom<ExplicitCtrFromTupleRef>> t2{t1};
    assert(!nonConstCopyCtrCalled(cuda::std::get<0>(t2).v));
  }
#endif // !TEST_COMPILER(GCC, <, 13)
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
