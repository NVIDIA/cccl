//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_fmax(T value)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((T) 0, (T) 0)), T>::value), "");
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::fmax((float) 0, (T) 0)), cuda::std::__promote_t<float, T>>::value), "");
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::fmax((double) 0, (T) 0)), cuda::std::__promote_t<double, T>>::value), "");
  assert(cuda::std::fmax(value, (T) 0) == value);
}

__host__ __device__ void test_fmax(float value)
{
  test_fmax<float>(value);
  test_fmax<double>(value);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_fmax<long double>(value);
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_fmax<__half>(__float2half(value));
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_fmax<__nv_bfloat16>(__float2bfloat16(value));
#endif // _LIBCUDACXX_HAS_NVBF16

  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((int) 0, (int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((int) 0, (long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((int) 0, (unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((float) 0, (unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((double) 0, (long) 0)), double>::value), "");

  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((bool) 0, (float) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((unsigned short) 0, (double) 0)), double>::value), "");

  static_assert((cuda::std::is_same<decltype(cuda::std::fmaxf(0, 0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::fmax((long double) 0, (unsigned long) 0)), long double>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmaxl(0, 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

template <class T>
__host__ __device__ void test_fmin(T value)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((T) 0, (T) 0)), T>::value), "");
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::fmin((float) 0, (T) 0)), cuda::std::__promote_t<float, T>>::value), "");
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::fmin((double) 0, (T) 0)), cuda::std::__promote_t<double, T>>::value), "");
  assert(cuda::std::fmin(value, (T) 0) == T(0));
}

__host__ __device__ void test_fmin(float value)
{
  test_fmax<float>(value);
  test_fmax<double>(value);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_fmax<long double>(value);
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_fmax<__half>(__float2half(value));
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_fmax<__nv_bfloat16>(__float2bfloat16(value));
#endif // _LIBCUDACXX_HAS_NVBF16

  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((int) 0, (int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((int) 0, (long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((int) 0, (unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((float) 0, (unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((double) 0, (long) 0)), double>::value), "");

  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((bool) 0, (float) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((unsigned short) 0, (double) 0)), double>::value), "");

  static_assert((cuda::std::is_same<decltype(cuda::std::fminf(0, 0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::fmin((long double) 0, (unsigned long) 0)), long double>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::fminl(0, 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
}

__host__ __device__ void test(float value)
{
  test_fmax(value);
  test_fmin(value);
}

__global__ void test_global_kernel(float* value)
{
  test(*value);
}

int main(int, char**)
{
  volatile float value = 1.0f;
  test(value);
  return 0;
}
