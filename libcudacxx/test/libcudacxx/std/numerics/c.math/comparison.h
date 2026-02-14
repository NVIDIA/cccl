//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CMATH_COMPARISON_H
#define TEST_CMATH_COMPARISON_H

#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool eq(T lhs, T rhs) noexcept
{
  return lhs == rhs;
}

template <typename T, typename U, cuda::std::enable_if_t<cuda::std::is_arithmetic_v<U>, int> = 0>
__host__ __device__ constexpr bool eq(T lhs, U rhs) noexcept
{
  return eq(lhs, T(rhs));
}

#if _CCCL_HAS_NVFP16()
__host__ __device__ bool eq(__half lhs, __half rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__half2float(lhs) == ::__half2float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
__host__ __device__ bool eq(__nv_bfloat16 lhs, __nv_bfloat16 rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__bfloat162float(lhs) == ::__bfloat162float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVBF16()

template <class Integer, cuda::std::enable_if_t<cuda::std::is_integral_v<Integer>, int> = 0>
__host__ __device__ bool is_about(Integer x, Integer y)
{
  return true;
}

__host__ __device__ bool is_about(float x, float y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-6);
}

__host__ __device__ bool is_about(double x, double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}

#if _CCCL_HAS_LONG_DOUBLE()
__host__ __device__ bool is_about(long double x, long double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
__host__ __device__ bool is_about(__half x, __half y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __half(1e-3));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
__host__ __device__ bool is_about(__nv_bfloat16 x, __nv_bfloat16 y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __nv_bfloat16(5e-3));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#endif // TEST_CMATH_COMPARISON_H
