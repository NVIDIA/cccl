//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class U = T>
//   constexpr explicit(!is_convertible_v<U, T>) expected(U&& v);
//
// Constraints:
// - is_same_v<remove_cvref_t<U>, in_place_t> is false; and
// - is_same_v<expected, remove_cvref_t<U>> is false; and
// - remove_cvref_t<U> is not a specialization of unexpected; and
// - is_constructible_v<T, U> is true.
//
// Effects: Direct-non-list-initializes val with cuda::std::forward<U>(v).
//
// Postconditions: has_value() is true.
//
// Throws: Any exception thrown by the initialization of val.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, int>, "");

// is_same_v<remove_cvref_t<U>, in_place_t>
struct FromJustInplace
{
  __host__ __device__ FromJustInplace(cuda::std::in_place_t);
};
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<FromJustInplace, int>, cuda::std::in_place_t>, "");
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<FromJustInplace, int>, cuda::std::in_place_t const&>,
              "");

// is_same_v<expected, remove_cvref_t<U>>
// Note that result is true because it is covered by the constructors that take expected
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::expected<int, int>&>, "");

// remove_cvref_t<U> is a specialization of unexpected
// Note that result is true because it is covered by the constructors that take unexpected
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::unexpected<int>&>, "");

// !is_constructible_v<T, U>
struct foo
{};
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<int, int>, foo>, "");

// test explicit(!is_convertible_v<U, T>)
struct NotConvertible
{
  __host__ __device__ explicit NotConvertible(int);
};
static_assert(cuda::std::is_convertible_v<int, cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_convertible_v<int, cuda::std::expected<NotConvertible, int>>, "");

struct CopyOnly
{
  int i;
  __host__ __device__ constexpr CopyOnly(int ii)
      : i(ii)
  {}
  CopyOnly(const CopyOnly&)                = default;
  __host__ __device__ CopyOnly(CopyOnly&&) = delete;
  __host__ __device__ friend constexpr bool operator==(const CopyOnly& mi, int ii)
  {
    return mi.i == ii;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const CopyOnly& mi, int ii)
  {
    return mi.i != ii;
  }
#endif // TEST_STD_VER < 2020
};

template <class T>
__host__ __device__ constexpr void testInt()
{
  cuda::std::expected<T, int> e(5);
  assert(e.has_value());
  assert(e.value() == 5);
}

template <class T>
__host__ __device__ constexpr void testLValue()
{
  T t(5);
  cuda::std::expected<T, int> e(t);
  assert(e.has_value());
  assert(e.value() == 5);
}

template <class T>
__host__ __device__ constexpr void testRValue()
{
  cuda::std::expected<T, int> e(T(5));
  assert(e.has_value());
  assert(e.value() == 5);
}

__host__ __device__ constexpr bool test()
{
  testInt<int>();
  testInt<CopyOnly>();
  testInt<MoveOnly>();
  testLValue<int>();
  testLValue<CopyOnly>();
  testRValue<int>();
  testRValue<MoveOnly>();

  // Test default template argument.
  // Without it, the template parameter cannot be deduced from an initializer list
  {
    struct Bar
    {
      int i;
      int j;
      __host__ __device__ constexpr Bar(int ii, int jj)
          : i(ii)
          , j(jj)
      {}
    };

    cuda::std::expected<Bar, int> e({5, 6});
    assert(e.value().i == 5);
    assert(e.value().j == 6);
  }

  // this is a confusing example, but the behaviour
  // is exactly what is specified in the spec
  // see https://cplusplus.github.io/LWG/issue3836
  {
    struct BaseError
    {};
    struct DerivedError : BaseError
    {};

    cuda::std::expected<bool, DerivedError> e1(false);
    cuda::std::expected<bool, BaseError> e2(e1);
    assert(e2.has_value());
    assert(e2.value()); // yes, e2 holds "true"
  }
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  struct Except
  {};

  struct Throwing
  {
    __host__ __device__ Throwing(int)
    {
      throw Except{};
    };
  };

  try
  {
    cuda::std::expected<Throwing, int> u(5);
    assert(false);
  }
  catch (Except)
  {}
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
