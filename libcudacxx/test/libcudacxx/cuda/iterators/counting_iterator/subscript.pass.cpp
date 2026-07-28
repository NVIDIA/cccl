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
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

template <class Iter>
TEST_FUNC constexpr void testType()
{
  using T         = typename Iter::value_type;
  using diff_type = typename Iter::difference_type;
  {
    Iter iter{T{0}};
    for (diff_type i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }
  {
    Iter iter{T{10}};
    for (diff_type i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }

  {
    const Iter iter{T{0}};
    for (diff_type i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }
  {
    const Iter iter{T{10}};
    for (diff_type i = 0; i < 100; ++i)
    {
      assert(iter[i] == T(i + 10));
    }
    static_assert(noexcept(iter[0]) == !cuda::std::same_as<T, SomeInt>);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), T>);
  }
}

TEST_FUNC constexpr bool test()
{
  testType<cuda::counting_iterator<SomeInt>>();
  testType<cuda::counting_iterator<SomeInt, cuda::std::int16_t>>();

  testType<cuda::counting_iterator<signed long>>();
  testType<cuda::counting_iterator<signed long, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned long>>();
  testType<cuda::counting_iterator<unsigned long, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<int>>();
  testType<cuda::counting_iterator<int, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned>>();
  testType<cuda::counting_iterator<unsigned, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<short>>();
  testType<cuda::counting_iterator<short, cuda::std::int8_t>>();

  testType<cuda::counting_iterator<unsigned short>>();
  testType<cuda::counting_iterator<unsigned short, cuda::std::int8_t>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
