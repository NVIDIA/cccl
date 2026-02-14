//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator(Fn);
// constexpr explicit iterator(Fn, Integer);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class InputFn, class OutputFn>
__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  InputFn input_func{};
  OutputFn output_func{};

  { // CTAD
    cuda::transform_input_output_iterator iter{random_access_iterator{buffer + 2}, input_func, output_func};
    assert(base(iter.base()) == buffer + 2);
    assert(*iter == input_func(buffer[2]));
    *iter = 3;
    assert(buffer[2] == output_func(3));
    buffer[2] = 2;

    // The test iterators are not `is_nothrow_move_constructible`
#if !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    static_assert(
      !noexcept(cuda::transform_input_output_iterator{random_access_iterator{buffer + 2}, input_func, output_func}));
#endif // !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    static_assert(
      cuda::std::is_same_v<decltype(iter),
                           cuda::transform_input_output_iterator<InputFn, OutputFn, random_access_iterator<int*>>>);
  }

  { // CTAD
    cuda::transform_input_output_iterator iter{buffer + 2, input_func, output_func};
    assert(iter.base() == buffer + 2);
    assert(*iter == input_func(buffer[2]));
    *iter = 3;
    assert(buffer[2] == output_func(3));
    buffer[2] = 2;

    static_assert(noexcept(cuda::transform_input_output_iterator{buffer + 2, input_func, output_func}));
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::transform_input_output_iterator<InputFn, OutputFn, int*>>);
  }

  {
    cuda::transform_input_output_iterator<InputFn, OutputFn, random_access_iterator<int*>> iter{
      random_access_iterator{buffer + 2}, input_func, output_func};
    assert(base(iter.base()) == buffer + 2);
    assert(*iter == input_func(buffer[2]));
    *iter = 3;
    assert(buffer[2] == output_func(3));
    buffer[2] = 2;

#if !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    // The test iterators are not `is_nothrow_move_constructible`
    static_assert(!noexcept(cuda::transform_input_output_iterator<InputFn, OutputFn, random_access_iterator<int*>>{
      random_access_iterator{buffer + 2}, input_func, output_func}));
#endif // !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
  }

  {
    cuda::transform_input_output_iterator<InputFn, OutputFn, int*> iter{buffer + 2, input_func, output_func};
    assert(iter.base() == buffer + 2);
    assert(*iter == input_func(buffer[2]));
    *iter = 3;
    assert(buffer[2] == output_func(3));

    static_assert(
      noexcept(cuda::transform_input_output_iterator<InputFn, OutputFn, int*>{buffer + 2, input_func, output_func}));
  }

  return true;
}

__host__ __device__ constexpr bool test()
{
  test<PlusOne, TimesTwo>();
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test<PlusOneHost, TimesTwo>();), (test<PlusOneDevice, TimesTwo>();))

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
