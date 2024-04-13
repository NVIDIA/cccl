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
//   constexpr explicit unexpected(in_place_t, Args&&... args);
//
// Constraints: is_constructible_v<E, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<Args>(args)....
//
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

// Test Constraints:
static_assert(cuda::std::constructible_from<cuda::std::unexpected<int>, cuda::std::in_place_t, int>, "");

// !is_constructible_v<E, Args...>
struct Foo
{};
static_assert(!cuda::std::constructible_from<cuda::std::unexpected<Foo>, cuda::std::in_place_t, int>, "");

// test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(ImplicitlyConstructible_,
                             requires(Args&&... args)((conversion_test<T>({cuda::std::forward<Args>(args)...}))));

template <class T, class... Args>
constexpr bool ImplicitlyConstructible = _LIBCUDACXX_FRAGMENT(ImplicitlyConstructible_, T, Args...);

static_assert(ImplicitlyConstructible<int, int>, "");
static_assert(!ImplicitlyConstructible<cuda::std::unexpected<int>, cuda::std::in_place_t, int>, "");

struct Arg
{
  int i;
  __host__ __device__ constexpr Arg(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr Arg(const Arg& other)
      : i(other.i)
  {}
  __host__ __device__ constexpr Arg(Arg&& other)
      : i(other.i)
  {
    other.i = 0;
  }
};

struct Error
{
  Arg arg;
  __host__ __device__ constexpr explicit Error(const Arg& a)
      : arg(a)
  {}
  __host__ __device__ constexpr explicit Error(Arg&& a)
      : arg(cuda::std::move(a))
  {}
  __host__ __device__ Error(cuda::std::initializer_list<Error>)
      : arg(0)
  {
    assert(false);
  }
};

__host__ __device__ constexpr bool test()
{
  // lvalue
  {
    Arg a{5};
    cuda::std::unexpected<Error> unex(cuda::std::in_place, a);
    assert(unex.error().arg.i == 5);
    assert(a.i == 5);
  }

  // rvalue
  {
    Arg a{5};
    cuda::std::unexpected<Error> unex(cuda::std::in_place, cuda::std::move(a));
    assert(unex.error().arg.i == 5);
    assert(a.i == 0);
  }

  // Direct-non-list-initializes: does not trigger initializer_list overload
  {
    Error e(5);
    cuda::std::unexpected<Error> unex(cuda::std::in_place, e);
    unused(e);
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
    Throwing(int)
    {
      throw Except{};
    }
  };

  try
  {
    cuda::std::unexpected<Throwing> u(cuda::std::in_place, 5);
    assert(false);
  }
  catch (Except)
  {}
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
  static_assert(test(), "");
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
