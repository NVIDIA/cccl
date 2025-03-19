//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__ T euclidian_addition(T rhs, T lhs)
{
  const T modulus = maxValue - minValue + 1;
  T ret           = rhs + lhs;
  if (ret > maxValue)
  {
    ret -= modulus;
  }
  return ret;
}

//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__ T euclidian_subtraction(T lhs, T rhs)
{
  const T modulus = maxValue - minValue + 1;
  T ret           = lhs - rhs;
  if constexpr (cuda::std::is_signed_v<T>) // avoids warning about comparison with zero if T is unsigned
  {
    if (ret < minValue)
    {
      ret += modulus;
    }
  }
  if (ret > maxValue) // this can happen if T is unsigned
  {
    ret += modulus;
  }
  return ret;
}
