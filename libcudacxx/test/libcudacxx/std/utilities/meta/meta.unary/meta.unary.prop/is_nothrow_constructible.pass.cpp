//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_nothrow_constructible;

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC void test_is_nothrow_constructible()
{
  static_assert((cuda::std::is_nothrow_constructible<T>::value));
  static_assert((cuda::std::is_nothrow_constructible_v<T>) );
}

template <class T, class A0>
TEST_FUNC void test_is_nothrow_constructible()
{
  static_assert((cuda::std::is_nothrow_constructible<T, A0>::value));
  static_assert((cuda::std::is_nothrow_constructible_v<T, A0>) );
}

template <class T>
TEST_FUNC void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T>::value));
  static_assert((!cuda::std::is_nothrow_constructible_v<T>) );
}

template <class T, class A0>
TEST_FUNC void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T, A0>::value));
  static_assert((!cuda::std::is_nothrow_constructible_v<T, A0>) );
}

template <class T, class A0, class A1>
TEST_FUNC void test_is_not_nothrow_constructible()
{
  static_assert((!cuda::std::is_nothrow_constructible<T, A0, A1>::value));
  static_assert((!cuda::std::is_nothrow_constructible_v<T, A0, A1>) );
}

class Empty
{};

class NotEmpty
{
  TEST_FUNC virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  TEST_FUNC virtual ~Abstract() = 0;
};

struct A
{
  TEST_FUNC A(const A&);
};

struct C
{
  TEST_FUNC C(C&); // not const
  TEST_FUNC void operator=(C&); // not const
};

struct Tuple
{
  TEST_FUNC Tuple(Empty&&) noexcept {}
};

struct ConvertibleToInt
{
  TEST_FUNC constexpr operator int() const
  {
    return 42;
  }
};

struct NothrowConvertibleToInt
{
  TEST_FUNC constexpr operator int() const noexcept
  {
    return 42;
  }
};

TEST_FUNC void test_is_nothrow_constructible_only_conversion()
{
  {
    static_assert(cuda::std::is_constructible_v<int, ConvertibleToInt>);
    static_assert(!cuda::std::is_nothrow_constructible<int, ConvertibleToInt>::value);
    static_assert(!cuda::std::is_nothrow_constructible_v<int, ConvertibleToInt>);
  }
  {
    static_assert(cuda::std::is_constructible_v<int, NothrowConvertibleToInt>);
    static_assert(cuda::std::is_nothrow_constructible<int, NothrowConvertibleToInt>::value);
    static_assert(cuda::std::is_nothrow_constructible_v<int, NothrowConvertibleToInt>);
  }
}

int main(int, char**)
{
  test_is_nothrow_constructible<int>();
  test_is_nothrow_constructible<int, const int&>();
  test_is_nothrow_constructible<Empty>();
  test_is_nothrow_constructible<Empty, const Empty&>();

  test_is_not_nothrow_constructible<A, int>();
  test_is_not_nothrow_constructible<A, int, double>();
  test_is_not_nothrow_constructible<A>();
  test_is_not_nothrow_constructible<C>();
  test_is_nothrow_constructible<Tuple&&, Empty>(); // See bug #19616.

  static_assert(!cuda::std::is_constructible<Tuple&, Empty>::value);
  test_is_not_nothrow_constructible<Tuple&, Empty>(); // See bug #19616.

  // conversion only types
  test_is_nothrow_constructible_only_conversion();

  return 0;
}
