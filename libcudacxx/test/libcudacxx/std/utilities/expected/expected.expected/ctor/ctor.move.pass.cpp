//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr expected(expected&& rhs) noexcept(see below);
//
// Constraints:
// - is_move_constructible_v<T> is true and
// - is_move_constructible_v<E> is true.
//
// Effects: If rhs.has_value() is true, direct-non-list-initializes val with cuda::std::move(*rhs).
// Otherwise, direct-non-list-initializes unex with cuda::std::move(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of val or unex.
//
// Remarks: The exception specification is equivalent to is_nothrow_move_constructible_v<T> &&
// is_nothrow_move_constructible_v<E>.
//
// This constructor is trivial if
// - is_trivially_move_constructible_v<T> is true and
// - is_trivially_move_constructible_v<E> is true.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct NonMovable
{
  NonMovable(NonMovable&&) = delete;
};

struct MovableNonTrivial
{
  int i;
  __host__ __device__ constexpr MovableNonTrivial(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr MovableNonTrivial(MovableNonTrivial&& o)
      : i(o.i)
  {
    o.i = 0;
  }
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MovableNonTrivial&, const MovableNonTrivial&) = default;
#else
  __host__ __device__ friend constexpr bool
  operator==(const MovableNonTrivial& lhs, const MovableNonTrivial& rhs) noexcept
  {
    return lhs.i == rhs.i;
  };
  __host__ __device__ friend constexpr bool
  operator!=(const MovableNonTrivial& lhs, const MovableNonTrivial& rhs) noexcept
  {
    return lhs.i != rhs.i;
  };
#endif // TEST_STD_VER > 2017
};

struct MoveMayThrow
{
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) {}
};

// Test Constraints:
// - is_move_constructible_v<T> is true and
// - is_move_constructible_v<E> is true.
static_assert(cuda::std::is_move_constructible_v<cuda::std::expected<int, int>>, "");
static_assert(cuda::std::is_move_constructible_v<cuda::std::expected<MovableNonTrivial, int>>, "");
static_assert(cuda::std::is_move_constructible_v<cuda::std::expected<int, MovableNonTrivial>>, "");
static_assert(cuda::std::is_move_constructible_v<cuda::std::expected<MovableNonTrivial, MovableNonTrivial>>, "");
static_assert(!cuda::std::is_move_constructible_v<cuda::std::expected<NonMovable, int>>, "");
static_assert(!cuda::std::is_move_constructible_v<cuda::std::expected<int, NonMovable>>, "");
static_assert(!cuda::std::is_move_constructible_v<cuda::std::expected<NonMovable, NonMovable>>, "");

// Test: This constructor is trivial if
// - is_trivially_move_constructible_v<T> is true and
// - is_trivially_move_constructible_v<E> is true.
static_assert(cuda::std::is_trivially_move_constructible_v<cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_trivially_move_constructible_v<cuda::std::expected<MovableNonTrivial, int>>, "");
static_assert(!cuda::std::is_trivially_move_constructible_v<cuda::std::expected<int, MovableNonTrivial>>, "");
static_assert(!cuda::std::is_trivially_move_constructible_v<cuda::std::expected<MovableNonTrivial, MovableNonTrivial>>,
              "");

#ifndef TEST_COMPILER_ICC
// Test: The exception specification is equivalent to
// is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>.
static_assert(cuda::std::is_nothrow_move_constructible_v<cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_nothrow_move_constructible_v<cuda::std::expected<MoveMayThrow, int>>, "");
static_assert(!cuda::std::is_nothrow_move_constructible_v<cuda::std::expected<int, MoveMayThrow>>, "");
static_assert(!cuda::std::is_nothrow_move_constructible_v<cuda::std::expected<MoveMayThrow, MoveMayThrow>>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // move the value non-trivial
  {
    cuda::std::expected<MovableNonTrivial, int> e1(5);
    auto e2 = cuda::std::move(e1);
    assert(e2.has_value());
    assert(e2.value().i == 5);
    assert(e1.has_value());
    assert(e1.value().i == 0);
  }

  // move the error non-trivial
  {
    cuda::std::expected<int, MovableNonTrivial> e1(cuda::std::unexpect, 5);
    auto e2 = cuda::std::move(e1);
    assert(!e2.has_value());
    assert(e2.error().i == 5);
    assert(!e1.has_value());
    assert(e1.error().i == 0);
  }

  // move the value trivial
  {
    cuda::std::expected<int, int> e1(5);
    auto e2 = cuda::std::move(e1);
    assert(e2.has_value());
    assert(e2.value() == 5);
    assert(e1.has_value());
  }

  // move the error trivial
  {
    cuda::std::expected<int, int> e1(cuda::std::unexpect, 5);
    auto e2 = cuda::std::move(e1);
    assert(!e2.has_value());
    assert(e2.error() == 5);
    assert(!e1.has_value());
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
    Throwing() = default;
    __host__ __device__ Throwing(Throwing&&)
    {
      throw Except{};
    }
  };

  // throw on moving value
  {
    cuda::std::expected<Throwing, int> e1;
    try
    {
      auto e2 = cuda::std::move(e1);
      unused(e2);
      assert(false);
    }
    catch (Except)
    {}
  }

  // throw on moving error
  {
    cuda::std::expected<int, Throwing> e1(cuda::std::unexpect);
    try
    {
      auto e2 = cuda::std::move(e1);
      unused(e2);
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
