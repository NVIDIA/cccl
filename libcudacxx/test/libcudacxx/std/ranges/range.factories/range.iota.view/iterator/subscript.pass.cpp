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

// constexpr W operator[](difference_type n) const
//   requires advanceable<W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  {
    cuda::std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
  }
  {
    cuda::std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
  }
  {
    const cuda::std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
  }
}

__host__ __device__ constexpr bool test()
{
  testType<SomeInt>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
