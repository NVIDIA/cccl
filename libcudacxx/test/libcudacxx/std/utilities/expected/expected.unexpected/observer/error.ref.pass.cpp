//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr E& error() & noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class = void>
constexpr bool ErrorNoexcept = false;

template <class T>
constexpr bool ErrorNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T&>().error())>> =
  noexcept(cuda::std::declval<T&>().error());

static_assert(!ErrorNoexcept<int>, "");
static_assert(ErrorNoexcept<cuda::std::unexpected<int>>, "");

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<int> unex(5);
  decltype(auto) i = unex.error();
  static_assert(cuda::std::same_as<decltype(i), int&>, "");
  assert(i == 5);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
