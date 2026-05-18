// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Some early versions (cl.exe 14.16 / VC141) do not identify correct constructors
// UNSUPPORTED: msvc

// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

// <cuda/std/tuple>

// template <class TupleLike> tuple(TupleLike&&); // libc++ extension

// See llvm.org/PR31384
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int count = 0;

struct Explicit
{
  Explicit() = default;
  TEST_FUNC explicit Explicit(int) {}
};

struct Implicit
{
  Implicit() = default;
  TEST_FUNC Implicit(int) {}
};

template <class T>
struct Derived : cuda::std::tuple<T>
{
  using cuda::std::tuple<T>::tuple;
  template <class U>
  TEST_FUNC operator cuda::std::tuple<U>() &&
  {
    ++count;
    return cuda::std::tuple<U>{};
  }
};

template <class T>
struct ExplicitDerived : cuda::std::tuple<T>
{
  using cuda::std::tuple<T>::tuple;
  template <class U>
  TEST_FUNC explicit operator cuda::std::tuple<U>() &&
  {
    ++count;
    return cuda::std::tuple<U>{};
  }
};

int main(int, char**)
{
  {
    [[maybe_unused]] cuda::std::tuple<Explicit> foo = Derived<int>{42};
    assert(count == 1);
    [[maybe_unused]] cuda::std::tuple<Explicit> bar(Derived<int>{42});
    NV_IF_ELSE_TARGET(NV_IS_HOST, (assert(count == 2);), (assert(count == 1);)) // nvbug6202272
  }
  count = 0;
  {
    [[maybe_unused]] cuda::std::tuple<Implicit> foo = Derived<int>{42};
    assert(count == 1);
    [[maybe_unused]] cuda::std::tuple<Implicit> bar(Derived<int>{42});
    NV_IF_ELSE_TARGET(NV_IS_HOST, (assert(count == 2);), (assert(count == 1);)) // nvbug6202272
  }
  count = 0;
  {
    static_assert(!cuda::std::is_convertible<ExplicitDerived<int>, cuda::std::tuple<Explicit>>::value);
    [[maybe_unused]] cuda::std::tuple<Explicit> bar(ExplicitDerived<int>{42});
    NV_IF_ELSE_TARGET(NV_IS_HOST, (assert(count == 1);), (assert(count == 0);)) // nvbug6202272
  }
  count = 0;
  {
    [[maybe_unused]] cuda::std::tuple<Implicit> foo = ExplicitDerived<int>{42};
    static_assert(cuda::std::is_convertible_v<ExplicitDerived<int>, cuda::std::tuple<Implicit>>);
    assert(count == 0);
    ExplicitDerived<int> d{42};
    [[maybe_unused]] cuda::std::tuple<Implicit> bar(cuda::std::move(d));
    NV_IF_ELSE_TARGET(NV_IS_HOST, (assert(count == 1);), (assert(count == 0);)) // nvbug6202272
  }
  count = 0;

  return 0;
}
