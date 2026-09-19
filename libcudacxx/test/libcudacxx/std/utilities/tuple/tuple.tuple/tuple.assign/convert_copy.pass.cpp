//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   tuple& operator=(const tuple<UTypes...>& u);

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct B
{
  int id_;

  TEST_FUNC constexpr explicit B(int i = 0)
      : id_(i)
  {}
};

struct D : B
{
  TEST_FUNC constexpr explicit D(int i = 0)
      : B(i)
  {}
};

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&)      = delete;
};

struct NothrowCopyAssignable
{
  NothrowCopyAssignable(NothrowCopyAssignable const&) = delete;
  TEST_FUNC constexpr NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept
  {
    return *this;
  }
};

struct PotentiallyThrowingCopyAssignable
{
  PotentiallyThrowingCopyAssignable(PotentiallyThrowingCopyAssignable const&) = delete;
  TEST_FUNC constexpr PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&)
  {
    return *this;
  }
};

TEST_FUNC constexpr bool test()
{
  {
    using T0 = cuda::std::tuple<long>;
    using T1 = cuda::std::tuple<long long>;
    T0 t0(2);
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    using T0 = cuda::std::tuple<long, char>;
    using T1 = cuda::std::tuple<long long, int>;
    T0 t0(2, 'a');
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
  }
  {
    using T0 = cuda::std::tuple<long, char, D>;
    using T1 = cuda::std::tuple<long long, int, B>;
    T0 t0(2, 'a', D(3));
    T1 t1;
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 3);
  }
  {
    D d(3);
    D d2(2);
    using T0 = cuda::std::tuple<long, char, D&>;
    using T1 = cuda::std::tuple<long long, int, B&>;
    T0 t0(2, 'a', d2);
    T1 t1(1, 'b', d);
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 2);
  }
  {
    // Test that tuple evaluates correctly applies an lvalue reference
    // before evaluating is_assignable (i.e. 'is_assignable<int&, int&>')
    // instead of evaluating 'is_assignable<int&&, int&>' which is false.
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&&> t(cuda::std::move(x));
    cuda::std::tuple<int&> t2(y);
    t = t2;
    assert(cuda::std::get<0>(t) == 43);
    assert(&cuda::std::get<0>(t) == &x);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  {
    using T = cuda::std::tuple<int, NonAssignable>;
    using U = cuda::std::tuple<NonAssignable, int>;
    static_assert(!cuda::std::is_assignable<T&, U const&>::value);
    static_assert(!cuda::std::is_assignable<U&, T const&>::value);
  }
  {
    using T0 = cuda::std::tuple<NothrowCopyAssignable, long>;
    using T1 = cuda::std::tuple<NothrowCopyAssignable, int>;
    static_assert(cuda::std::is_nothrow_assignable<T0&, T1 const&>::value);
  }
  {
    using T0 = cuda::std::tuple<PotentiallyThrowingCopyAssignable, long>;
    using T1 = cuda::std::tuple<PotentiallyThrowingCopyAssignable, int>;
    static_assert(cuda::std::is_assignable<T0&, T1 const&>::value);
    static_assert(!cuda::std::is_nothrow_assignable<T0&, T1 const&>::value);
  }

  return 0;
}
