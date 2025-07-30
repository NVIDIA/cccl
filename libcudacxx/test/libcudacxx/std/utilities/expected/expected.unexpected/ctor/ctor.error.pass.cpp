//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class Err = E>
//   constexpr explicit unexpected(Err&& e);
//
// Constraints:
// - is_same_v<remove_cvref_t<Err>, unexpected> is false; and
// - is_same_v<remove_cvref_t<Err>, in_place_t> is false; and
// - is_constructible_v<E, Err> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<Err>(e).
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::constructible_from<cuda::std::unexpected<int>, int>, "");

// is_same_v<remove_cvref_t<Err>, unexpected>
struct CstrFromUnexpected
{
  __host__ __device__ CstrFromUnexpected(CstrFromUnexpected const&) = delete;
  __host__ __device__ CstrFromUnexpected(cuda::std::unexpected<CstrFromUnexpected> const&);
};
static_assert(
  !cuda::std::constructible_from<cuda::std::unexpected<CstrFromUnexpected>, cuda::std::unexpected<CstrFromUnexpected>>,
  "");

// is_same_v<remove_cvref_t<Err>, in_place_t>
struct CstrFromInplace
{
  __host__ __device__ CstrFromInplace(cuda::std::in_place_t);
};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<CstrFromInplace>, cuda::std::in_place_t>, "");

// !is_constructible_v<E, Err>
struct Foo
{};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<Foo>, int>, "");

// test explicit
static_assert(cuda::std::convertible_to<int, int>, "");
static_assert(!cuda::std::convertible_to<int, cuda::std::unexpected<int>>, "");

struct Error
{
  int i;
  __host__ __device__ constexpr Error(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr Error(const Error& other)
      : i(other.i)
  {}
  __host__ __device__ constexpr Error(Error&& other)
      : i(other.i)
  {
    other.i = 0;
  }
  __host__ __device__ Error(cuda::std::initializer_list<Error>)
  {
    assert(false);
  }
};

__host__ __device__ constexpr bool test()
{
  // lvalue
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(e);
    assert(unex.error().i == 5);
    assert(e.i == 5);
  }

  // rvalue
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(cuda::std::move(e));
    assert(unex.error().i == 5);
    assert(e.i == 0);
  }

  // Direct-non-list-initializes: does not trigger initializer_list overload
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(e);
    unused(e);
  }

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
    cuda::std::unexpected<Bar> ue({5, 6});
    assert(ue.error().i == 5);
    assert(ue.error().j == 6);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  struct Except
  {};

  struct Throwing
  {
    Throwing() = default;
    Throwing(const Throwing&)
    {
      throw Except{};
    }
  };

  Throwing t;
  try
  {
    cuda::std::unexpected<Throwing> u(t);
    assert(false);
  }
  catch (Except)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test(), "");
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
