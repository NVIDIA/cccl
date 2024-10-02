//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class... Args>
//   constexpr explicit expected(unexpect_t, Args&&... args);
//
// Constraints: is_constructible_v<E, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<Args>(args)....
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::unexpect_t>, "");
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::unexpect_t, int>, "");

// !is_constructible_v<T, Args...>
struct foo
{};
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<int, foo>, cuda::std::unexpect_t, int>, "");

// test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(ImplicitlyConstructible_,
                             requires(Args&&... args)((conversion_test<T>({cuda::std::forward<Args>(args)...}))));

template <class T, class... Args>
constexpr bool ImplicitlyConstructible = _LIBCUDACXX_FRAGMENT(ImplicitlyConstructible_, T, Args...);
static_assert(ImplicitlyConstructible<int, int>, "");

static_assert(!ImplicitlyConstructible<cuda::std::expected<int, int>, cuda::std::unexpect_t>, "");
static_assert(!ImplicitlyConstructible<cuda::std::expected<int, int>, cuda::std::unexpect_t, int>, "");

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
  cuda::std::expected<int, T> e(cuda::std::unexpect, 5);
  assert(!e.has_value());
  assert(e.error() == 5);
}

template <class T>
__host__ __device__ constexpr void testLValue()
{
  T t(5);
  cuda::std::expected<int, T> e(cuda::std::unexpect, t);
  assert(!e.has_value());
  assert(e.error() == 5);
}

template <class T>
__host__ __device__ constexpr void testRValue()
{
  cuda::std::expected<int, T> e(cuda::std::unexpect, T(5));
  assert(!e.has_value());
  assert(e.error() == 5);
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

  // no arg
  {
    cuda::std::expected<int, int> e(cuda::std::unexpect);
    assert(!e.has_value());
    assert(e.error() == 0);
  }

  // one arg
  {
    cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    assert(!e.has_value());
    assert(e.error() == 5);
  }

  // multi args
  {
    cuda::std::expected<int, cuda::std::tuple<int, short, MoveOnly>> e(cuda::std::unexpect, 1, short{2}, MoveOnly(3));
    assert(!e.has_value());
    assert((e.error() == cuda::std::tuple<int, short, MoveOnly>(1, short{2}, MoveOnly(3))));
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Except
{};
struct Throwing
{
  __host__ __device__ Throwing(int)
  {
    throw Except{};
  };
};
void test_exceptions()
{
  try
  {
    cuda::std::expected<int, Throwing> u(cuda::std::unexpect, 5);
    assert(false);
  }
  catch (const Except&)
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
