//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  A set of routines for testing the comparison operators of a type
//
//      FooOrder<expected-ordering>  All seven comparison operators, requires C++20 or newer.
//      FooComparison                All six pre-C++20 comparison operators
//      FooEquality                  Equality operators operator== and operator!=
//
//      AssertXAreNoexcept           static_asserts that the operations are all noexcept.
//      AssertXReturnBool            static_asserts that the operations return bool.
//      AssertOrderReturn            static_asserts that the pre-C++20 comparison operations
//                                   return bool and operator<=> returns the proper type.
//      AssertXConvertibleToBool     static_asserts that the operations return something convertible to bool.
//      testXValues                  returns the result of the comparison of all operations.
//
//      AssertOrderConvertibleToBool doesn't exist yet. It will be implemented when needed.

#ifndef TEST_COMPARISONS_H
#define TEST_COMPARISONS_H

#include <cuda/std/cassert>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Test the consistency of the six basic comparison operators for values that are ordered or unordered.
template <class T, class U = T>
[[nodiscard]] __host__ __device__ constexpr bool
testComparisonsComplete(const T& t1, const U& t2, bool isEqual, bool isLess, bool isGreater)
{
  assert(((isEqual ? 1 : 0) + (isLess ? 1 : 0) + (isGreater ? 1 : 0) <= 1)
         && "at most one of isEqual, isLess, and isGreater can be true");
  if (isEqual)
  {
    if (!(t1 == t2))
    {
      return false;
    }
    if (!(t2 == t1))
    {
      return false;
    }
    if ((t1 != t2))
    {
      return false;
    }
    if ((t2 != t1))
    {
      return false;
    }
    if ((t1 < t2))
    {
      return false;
    }
    if ((t2 < t1))
    {
      return false;
    }
    if (!(t1 <= t2))
    {
      return false;
    }
    if (!(t2 <= t1))
    {
      return false;
    }
    if ((t1 > t2))
    {
      return false;
    }
    if ((t2 > t1))
    {
      return false;
    }
    if (!(t1 >= t2))
    {
      return false;
    }
    if (!(t2 >= t1))
    {
      return false;
    }
  }
  else if (isLess)
  {
    if ((t1 == t2))
    {
      return false;
    }
    if ((t2 == t1))
    {
      return false;
    }
    if (!(t1 != t2))
    {
      return false;
    }
    if (!(t2 != t1))
    {
      return false;
    }
    if (!(t1 < t2))
    {
      return false;
    }
    if ((t2 < t1))
    {
      return false;
    }
    if (!(t1 <= t2))
    {
      return false;
    }
    if ((t2 <= t1))
    {
      return false;
    }
    if ((t1 > t2))
    {
      return false;
    }
    if (!(t2 > t1))
    {
      return false;
    }
    if ((t1 >= t2))
    {
      return false;
    }
    if (!(t2 >= t1))
    {
      return false;
    }
  }
  else if (isGreater)
  {
    if ((t1 == t2))
    {
      return false;
    }
    if ((t2 == t1))
    {
      return false;
    }
    if (!(t1 != t2))
    {
      return false;
    }
    if (!(t2 != t1))
    {
      return false;
    }
    if ((t1 < t2))
    {
      return false;
    }
    if (!(t2 < t1))
    {
      return false;
    }
    if ((t1 <= t2))
    {
      return false;
    }
    if (!(t2 <= t1))
    {
      return false;
    }
    if (!(t1 > t2))
    {
      return false;
    }
    if ((t2 > t1))
    {
      return false;
    }
    if (!(t1 >= t2))
    {
      return false;
    }
    if ((t2 >= t1))
    {
      return false;
    }
  }
  else
  { // unordered
    if ((t1 == t2))
    {
      return false;
    }
    if ((t2 == t1))
    {
      return false;
    }
    if (!(t1 != t2))
    {
      return false;
    }
    if (!(t2 != t1))
    {
      return false;
    }
    if ((t1 < t2))
    {
      return false;
    }
    if ((t2 < t1))
    {
      return false;
    }
    if ((t1 <= t2))
    {
      return false;
    }
    if ((t2 <= t1))
    {
      return false;
    }
    if ((t1 > t2))
    {
      return false;
    }
    if ((t2 > t1))
    {
      return false;
    }
    if ((t1 >= t2))
    {
      return false;
    }
    if ((t2 >= t1))
    {
      return false;
    }
  }

  return true;
}

// Test the six basic comparison operators for ordered values.
template <class T, class U = T>
[[nodiscard]] __host__ __device__ constexpr bool testComparisons(const T& t1, const U& t2, bool isEqual, bool isLess)
{
  assert(!(isEqual && isLess) && "isEqual and isLess cannot be both true");
  bool isGreater = !isEqual && !isLess;
  return testComparisonsComplete(t1, t2, isEqual, isLess, isGreater);
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
[[nodiscard]] __host__ __device__ constexpr bool testComparisonsValues(Param val1, Param val2)
{
  const bool isEqual   = val1 == val2;
  const bool isLess    = val1 < val2;
  const bool isGreater = val1 > val2;

  return testComparisonsComplete(T(val1), T(val2), isEqual, isLess, isGreater);
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertComparisonsAreNoexcept()
{
  static_assert(noexcept(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() < cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() <= cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() > cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() >= cuda::std::declval<const U&>()));
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertComparisonsReturnBool()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() < cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() <= cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() > cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() >= cuda::std::declval<const U&>()), bool>);
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertComparisonsConvertibleToBool()
{
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() < cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() <= cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() > cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() >= cuda::std::declval<const U&>()), bool>::value),
    "");
}

#if TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
template <class T, class U = T>
__host__ __device__ constexpr void AssertOrderAreNoexcept()
{
  AssertComparisonsAreNoexcept<T, U>();
  static_assert(noexcept(cuda::std::declval<const T&>() <=> cuda::std::declval<const U&>()));
}

template <class Order, class T, class U = T>
__host__ __device__ constexpr void AssertOrderReturn()
{
  AssertComparisonsReturnBool<T, U>();
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() <=> cuda::std::declval<const U&>()), Order>);
}

template <class Order, class T, class U = T>
[[nodiscard]] __host__ __device__ constexpr bool testOrder(const T& t1, const U& t2, Order order)
{
  bool equal   = order == Order::equivalent;
  bool less    = order == Order::less;
  bool greater = order == Order::greater;

  return (t1 <=> t2 == order) && testComparisonsComplete(t1, t2, equal, less, greater);
}

template <class T, class Param>
[[nodiscard]] __host__ __device__ constexpr bool testOrderValues(Param val1, Param val2)
{
  return testOrder(T(val1), T(val2), val1 <=> val2);
}

#endif // TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

//  Test all two comparison operations for sanity
template <class T, class U = T>
[[nodiscard]] __host__ __device__ constexpr bool testEquality(const T& t1, const U& t2, bool isEqual)
{
  if (isEqual)
  {
    if (!(t1 == t2))
    {
      return false;
    }
    if (!(t2 == t1))
    {
      return false;
    }
    if ((t1 != t2))
    {
      return false;
    }
    if ((t2 != t1))
    {
      return false;
    }
  }
  else /* not equal */
  {
    if ((t1 == t2))
    {
      return false;
    }
    if ((t2 == t1))
    {
      return false;
    }
    if (!(t1 != t2))
    {
      return false;
    }
    if (!(t2 != t1))
    {
      return false;
    }
  }

  return true;
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
[[nodiscard]] __host__ __device__ constexpr bool testEqualityValues(Param val1, Param val2)
{
  const bool isEqual = val1 == val2;

  return testEquality(T(val1), T(val2), isEqual);
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertEqualityAreNoexcept()
{
  static_assert(noexcept(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()));
  static_assert(noexcept(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()));
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertEqualityReturnBool()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()), bool>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()), bool>);
}

template <class T, class U = T>
__host__ __device__ constexpr void AssertEqualityConvertibleToBool()
{
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() == cuda::std::declval<const U&>()), bool>::value),
    "");
  static_assert(
    (cuda::std::is_convertible<decltype(cuda::std::declval<const T&>() != cuda::std::declval<const U&>()), bool>::value),
    "");
}

struct LessAndEqComp
{
  int value;

  __host__ __device__ constexpr LessAndEqComp(int v)
      : value(v)
  {}

  __host__ __device__ friend constexpr bool operator<(const LessAndEqComp& lhs, const LessAndEqComp& rhs)
  {
    return lhs.value < rhs.value;
  }

  __host__ __device__ friend constexpr bool operator==(const LessAndEqComp& lhs, const LessAndEqComp& rhs)
  {
    return lhs.value == rhs.value;
  }
};

#if TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
struct StrongOrder
{
  int value;
  __host__ __device__ constexpr StrongOrder(int v)
      : value(v)
  {}
  __host__ __device__ friend cuda::std::strong_ordering operator<=>(StrongOrder, StrongOrder) = default;
};

struct WeakOrder
{
  int value;
  __host__ __device__ constexpr WeakOrder(int v)
      : value(v)
  {}
  __host__ __device__ friend cuda::std::weak_ordering operator<=>(WeakOrder, WeakOrder) = default;
};

struct PartialOrder
{
  int value;
  __host__ __device__ constexpr PartialOrder(int v)
      : value(v)
  {}
  __host__ __device__ friend constexpr cuda::std::partial_ordering operator<=>(PartialOrder lhs, PartialOrder rhs)
  {
    if (lhs.value == cuda::std::numeric_limits<int>::min() || rhs.value == cuda::std::numeric_limits<int>::min())
    {
      return cuda::std::partial_ordering::unordered;
    }
    return lhs.value <=> rhs.value;
  }
  __host__ __device__ friend constexpr bool operator==(PartialOrder lhs, PartialOrder rhs)
  {
    return (lhs <=> rhs) == cuda::std::partial_ordering::equivalent;
  }
};

#endif // TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#endif // TEST_COMPARISONS_H
