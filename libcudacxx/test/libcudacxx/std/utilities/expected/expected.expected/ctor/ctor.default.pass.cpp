//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

//
// constexpr expected();

// Constraints: is_default_constructible_v<T> is true.
//
// Effects: Value-initializes val.
// Postconditions: has_value() is true.
//
// Throws: Any exception thrown by the initialization of val.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NoDedefaultCtor
{
  NoDedefaultCtor() = delete;
};

// Test constraints
static_assert(cuda::std::is_default_constructible_v<cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_default_constructible_v<cuda::std::expected<NoDedefaultCtor, int>>, "");

struct MyInt
{
  int i;
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
#endif // TEST_STD_VER > 2017
};

template <class T, class E>
__host__ __device__ constexpr void testDefaultCtor()
{
  cuda::std::expected<T, E> e;
  assert(e.has_value());
  assert(e.value() == T());
}

template <class T>
__host__ __device__ constexpr void testTypes()
{
  testDefaultCtor<T, int>();
  testDefaultCtor<T, NoDedefaultCtor>();
}

__host__ __device__ constexpr bool test()
{
  testTypes<int>();
  testTypes<MyInt>();
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Except
{};
struct Throwing
{
  Throwing()
  {
    throw Except{};
  };
};
void test_exceptions()
{
  try
  {
    cuda::std::expected<Throwing, int> u;
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
