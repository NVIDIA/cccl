//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class G>
//   constexpr explicit(!is_convertible_v<G, E>) expected(unexpected<G>&& e);
//
// Let GF be G
//
// Constraints: is_constructible_v<E, GF> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<GF>(e.error()).
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints
static_assert(cuda::std::is_constructible_v<cuda::std::expected<void, int>, cuda::std::unexpected<int>>, "");
static_assert(cuda::std::is_constructible_v<cuda::std::expected<void, MoveOnly>, cuda::std::unexpected<MoveOnly>>, "");

// !is_constructible_v<E, GF>
struct foo
{};
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<void, int>, cuda::std::unexpected<foo>>, "");

// explicit(!is_convertible_v<G, E>)
struct NotConvertible
{
  __host__ __device__ explicit NotConvertible(int);
};
static_assert(cuda::std::is_convertible_v<cuda::std::unexpected<int>&&, cuda::std::expected<void, int>>, "");
static_assert(!cuda::std::is_convertible_v<cuda::std::unexpected<int>&&, cuda::std::expected<void, NotConvertible>>,
              "");

struct MyInt
{
  int i;
  __host__ __device__ constexpr MyInt(int ii)
      : i(ii)
  {}
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  };
  __host__ __device__ friend constexpr bool operator!=(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i != rhs.i;
  };
#endif // TEST_STD_VER > 2017
};

template <class Err>
__host__ __device__ constexpr void testInt()
{
  cuda::std::unexpected<int> u(5);
  cuda::std::expected<void, Err> e(cuda::std::move(u));
  assert(!e.has_value());
  assert(e.error() == 5);
}

__host__ __device__ constexpr void testMoveOnly()
{
  cuda::std::unexpected<MoveOnly> u(MoveOnly(5));
  cuda::std::expected<void, MoveOnly> e(cuda::std::move(u));
  assert(!e.has_value());
  assert(e.error() == 5);
  assert(u.error() == 0);
}

__host__ __device__ constexpr bool test()
{
  testInt<int>();
  testInt<MyInt>();
  testInt<MoveOnly>();
  testMoveOnly();
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
    }
  };

  {
    cuda::std::unexpected<int> u(5);
    try
    {
      cuda::std::expected<void, Throwing> e(cuda::std::move(u));
      unused(e);
      assert(false);
    }
    catch (Except)
    {}
  }
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
