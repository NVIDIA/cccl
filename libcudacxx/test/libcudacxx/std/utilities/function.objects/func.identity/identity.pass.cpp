//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// struct identity;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "MoveOnly.h"
#include "test_macros.h"

static_assert(cuda::std::semiregular<cuda::std::identity>);

#if TEST_STD_VER > 2017
static_assert(requires { typename cuda::std::identity::is_transparent; });
#else
template <class, class = void>
constexpr bool is_transparent = false;

template <class T>
constexpr bool is_transparent<T, cuda::std::void_t<typename T::is_transparent>> = true;

static_assert(is_transparent<cuda::std::identity>);
#endif

__host__ __device__ constexpr bool test()
{
  cuda::std::identity id{};
  int i = 42;
  assert(id(i) == 42);
  assert(id(cuda::std::move(i)) == 42);

  MoveOnly m1 = 2;
  MoveOnly m2 = id(cuda::std::move(m1));
  assert(m2.get() == 2);

  assert(&id(i) == &i);
  static_assert(&id(id) == &id);

  const cuda::std::identity idc{};
  assert(idc(1) == 1);
  assert(cuda::std::move(id)(1) == 1);
  assert(cuda::std::move(idc)(1) == 1);

  id = idc; // run-time checks assignment
  static_assert(cuda::std::is_same_v<decltype(id(i)), int&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int&&>())), int&&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int const&>())), int const&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int const&&>())), int const&&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int volatile&>())), int volatile&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int volatile&&>())), int volatile&&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int const volatile&>())), int const volatile&>);
  static_assert(cuda::std::is_same_v<decltype(id(cuda::std::declval<int const volatile&&>())), int const volatile&&>);

  struct S
  {
    constexpr S() = default;
    __host__ __device__ constexpr S(S&&) noexcept(false) {}
    __host__ __device__ constexpr S(S const&) noexcept(false) {}
  };
  S x{};
  static_assert(noexcept(id(x)));
  static_assert(noexcept(id(S())));
  unused(x);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test());
#endif

  return 0;
}
