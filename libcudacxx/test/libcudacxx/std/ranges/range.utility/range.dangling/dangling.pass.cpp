//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::dangling;

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::is_empty_v<cuda::std::ranges::dangling>);

template <int>
struct S
{};
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>, S<1>>);
static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::dangling, S<0>, S<1>, S<2>>);

__host__ __device__ constexpr bool test_dangling()
{
  auto a = cuda::std::ranges::dangling();
  auto b = cuda::std::ranges::dangling(S<0>());
  auto c = cuda::std::ranges::dangling(S<0>(), S<1>());
  auto d = cuda::std::ranges::dangling(S<0>(), S<1>(), S<2>());
  unused(a);
  unused(b);
  unused(c);
  unused(d);
  return true;
}

int main(int, char**)
{
  static_assert(test_dangling());
  test_dangling();
  return 0;
}
