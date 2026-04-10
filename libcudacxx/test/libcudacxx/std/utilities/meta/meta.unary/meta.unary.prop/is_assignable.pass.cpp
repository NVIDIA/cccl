//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_assignable

#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

struct B
{
  TEST_FUNC void operator=(A);
};

template <class T, class U>
TEST_FUNC void test_is_assignable()
{
  static_assert((cuda::std::is_assignable<T, U>::value));
  static_assert(cuda::std::is_assignable_v<T, U>);
}

template <class T, class U>
TEST_FUNC void test_is_not_assignable()
{
  static_assert((!cuda::std::is_assignable<T, U>::value));
  static_assert(!cuda::std::is_assignable_v<T, U>);
}

struct D;

struct C
{
  template <class U>
  TEST_FUNC D operator,(U&&);
};

struct E
{
  TEST_FUNC C operator=(int);
};

template <typename T>
struct X
{
  T t;
};

int main(int, char**)
{
  test_is_assignable<int&, int&>();
  test_is_assignable<int&, int>();
  test_is_assignable<int&, double>();
  test_is_assignable<B, A>();
  test_is_assignable<void*&, void*>();

  test_is_assignable<E, int>();

  test_is_not_assignable<int, int&>();
  test_is_not_assignable<int, int>();
  test_is_not_assignable<A, B>();
  test_is_not_assignable<void, const void>();
  test_is_not_assignable<const void, const void>();
  test_is_not_assignable<int(), int>();

  //  pointer to incomplete template type
  test_is_assignable<X<D>*&, X<D>*>();

  return 0;
}
