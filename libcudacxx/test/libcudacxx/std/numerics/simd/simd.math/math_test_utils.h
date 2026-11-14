//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SIMD_MATH_TEST_UTILS_H
#define SIMD_MATH_TEST_UTILS_H

#include <cuda/std/cassert>

#include "test_macros.h"

template <typename V>
struct math_wrapper
{
  V value;

  TEST_HOST_DEVICE_FUNC constexpr operator V() const
  {
    return value;
  }

  TEST_HOST_DEVICE_FUNC friend constexpr V operator+(const math_wrapper& x, const math_wrapper& y)
  {
    return x.value + y.value;
  }
};

template <typename V>
struct explicit_math_wrapper
{
  V value;

  TEST_HOST_DEVICE_FUNC explicit constexpr operator V() const
  {
    return value;
  }

  TEST_HOST_DEVICE_FUNC friend constexpr V operator+(const explicit_math_wrapper& x, const explicit_math_wrapper& y)
  {
    return x.value + y.value;
  }
};

template <typename V, typename T>
TEST_HOST_DEVICE_FUNC constexpr void check_math_values(const V& values, T expected)
{
  for (int i = 0; i < V::size(); ++i)
  {
    assert(values[i] == expected);
  }
}

#endif // SIMD_MATH_TEST_UTILS_H
