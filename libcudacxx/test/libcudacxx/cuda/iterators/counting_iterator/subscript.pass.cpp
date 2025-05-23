//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr W operator[](difference_type n) const
//   requires advanceable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void testType()
{
  {
    cuda::counting_iterator<T> iter{T{0}};
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }
  {
    cuda::counting_iterator<T> iter{T{10}};
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }

  {
    const cuda::counting_iterator<T> iter{T{0}};
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }
  {
    const cuda::counting_iterator<T> iter{T{10}};
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
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
