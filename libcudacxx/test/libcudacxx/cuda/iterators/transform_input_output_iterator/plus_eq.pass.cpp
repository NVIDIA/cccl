//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator+=(difference_type n);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PlusOne input_func{};
  TimesTwo output_func{};

  cuda::transform_input_output_iterator iter1{buffer + 1, input_func, output_func};
  cuda::transform_input_output_iterator iter2{buffer + 1, input_func, output_func};
  assert(iter1 == iter2);
  iter1 += 0;
  assert(iter1.base() == buffer + 1);
  assert(iter1 == iter2);
  iter1 += 5;
  assert(iter1.base() == buffer + 6);
  assert(iter1 != iter2);
  assert(iter1 == (iter2 + 5));

  static_assert(noexcept(iter2 += 5));
  static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
