//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto operator[](difference_type n) const;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class InputFn, class OutputFn>
__host__ __device__ constexpr void test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  InputFn input_func{};
  OutputFn output_func{};

  {
    cuda::transform_input_output_iterator iter{buffer, input_func, output_func};
    for (int i = 0; i < 8; ++i)
    {
      assert(iter[i] == input_func(buffer[i]));
      iter[i] = i;
      assert(buffer[i] == output_func(i));
    }
    static_assert(noexcept(iter[0]));
    static_assert(noexcept(static_cast<int>(iter[0])) == !cuda::std::is_same_v<InputFn, TimesTwoMayThrow>);
    static_assert(noexcept(iter[0] = 2) == !cuda::std::is_same_v<OutputFn, PlusOneMayThrow>);
    static_assert(!cuda::std::is_same_v<decltype(iter[0]), int&>);
    static_assert(cuda::std::is_convertible_v<decltype(iter[0]), int>);
  }

  {
    const cuda::transform_input_output_iterator iter{buffer, input_func, output_func};
    assert(iter[2] == input_func(buffer[2]));
    iter[2] = 2;
    assert(buffer[2] == output_func(2));
    static_assert(noexcept(iter[0]));
    static_assert(noexcept(static_cast<int>(iter[0])) == !cuda::std::is_same_v<InputFn, TimesTwoMayThrow>);
    static_assert(noexcept(iter[0] = 2) == !cuda::std::is_same_v<OutputFn, PlusOneMayThrow>);
    static_assert(!cuda::std::is_same_v<decltype(iter[0]), int&>);
    static_assert(cuda::std::is_convertible_v<decltype(iter[0]), int>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<TimesTwo, PlusOne>();
  test<TimesTwo, PlusOneMutable>();
  test<TimesTwo, PlusOneMayThrow>();
  test<TimesTwoMayThrow, PlusOne>();
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test<TimesTwo, PlusOneHost>();), (test<TimesTwo, PlusOneDevice>();))

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
