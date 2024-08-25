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

// template <class _Tp>
// inline constexpr empty_view<_Tp> empty{};

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  ASSERT_SAME_TYPE(decltype(cuda::std::views::empty<T>), const cuda::std::ranges::empty_view<T>);
  ASSERT_SAME_TYPE(decltype((cuda::std::views::empty<T>) ), const cuda::std::ranges::empty_view<T>&);

  auto v = cuda::std::views::empty<T>;
  assert(cuda::std::ranges::empty(v));
}

struct Empty
{};
struct BigType
{
  char buff[8];
};

__host__ __device__ constexpr bool test()
{
  testType<int>();
  testType<const int>();
  testType<int*>();
  testType<Empty>();
  testType<const Empty>();
  testType<BigType>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
