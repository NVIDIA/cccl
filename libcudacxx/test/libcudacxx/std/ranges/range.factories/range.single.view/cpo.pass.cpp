//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::views::single

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Can't invoke without arguments.
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::single))>);
// Can't invoke with a move-only type.
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::single)), MoveOnly>);

__host__ __device__ constexpr bool test()
{
  // Lvalue.
  {
    int x            = 42;
    decltype(auto) v = cuda::std::views::single(x);
    static_assert(cuda::std::same_as<decltype(v), cuda::std::ranges::single_view<int>>);
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  // Prvalue.
  {
    decltype(auto) v = cuda::std::views::single(42);
    static_assert(cuda::std::same_as<decltype(v), cuda::std::ranges::single_view<int>>);
    assert(v.size() == 1);
    assert(v.front() == 42);
  }

  // Const lvalue.
  {
    const int x      = 42;
    decltype(auto) v = cuda::std::views::single(x);
    static_assert(cuda::std::same_as<decltype(v), cuda::std::ranges::single_view<int>>);
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  // Xvalue.
  {
    int x            = 42;
    decltype(auto) v = cuda::std::views::single(cuda::std::move(x));
    static_assert(cuda::std::same_as<decltype(v), cuda::std::ranges::single_view<int>>);
    assert(v.size() == 1);
    assert(v.front() == x);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
