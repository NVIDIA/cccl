//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(const tuple&) = default;
// tuple(tuple&&) = default;

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NonTrivialMutableCopy
{
  NonTrivialMutableCopy(const NonTrivialMutableCopy&) = default;
  TEST_FUNC NonTrivialMutableCopy(NonTrivialMutableCopy&) {}
};

struct NonTrivialConstMove
{
  NonTrivialConstMove(const NonTrivialConstMove&) = default;
  TEST_FUNC NonTrivialConstMove(const NonTrivialConstMove&&) {}
};

// tuple(tuple<UTypes...>&) and tuple(const tuple<UTypes...>&&) must not replace a trivial copy
template <class T>
TEST_FUNC constexpr void test_trivial()
{
  static_assert(cuda::std::is_trivially_copy_constructible_v<T>);
  static_assert(cuda::std::is_trivially_move_constructible_v<T>);
  static_assert(cuda::std::is_trivially_constructible_v<T, T&>);
  static_assert(cuda::std::is_trivially_constructible_v<T, const T&&>);
}

int main(int, char**)
{
  test_trivial<cuda::std::tuple<int>>();
  test_trivial<cuda::std::tuple<int, short>>();
  test_trivial<cuda::std::tuple<int, short, double>>();
  test_trivial<cuda::std::tuple<int&, int>>();
  test_trivial<cuda::std::tuple<const int&, int>>();
  {
    using T = cuda::std::tuple<int, NonTrivialMutableCopy>;
    static_assert(cuda::std::is_constructible_v<T, T&>);
    static_assert(!cuda::std::is_trivially_constructible_v<T, T&>);
  }
  {
    using T = cuda::std::tuple<int, NonTrivialConstMove>;
    static_assert(cuda::std::is_constructible_v<T, const T&&>);
    static_assert(!cuda::std::is_trivially_constructible_v<T, const T&&>);
  }
  // The copy constructor is deleted for an rvalue reference element, so a const rvalue needs tuple(const tuple<U...>&&)
  {
    using T = cuda::std::tuple<int&&, int>;
    static_assert(!cuda::std::is_copy_constructible_v<T>);
    static_assert(cuda::std::is_constructible_v<T, const T&&>);
  }

  return 0;
}
