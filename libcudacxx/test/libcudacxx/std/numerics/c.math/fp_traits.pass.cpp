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
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "fp_compare.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_fpclassify(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::fpclassify(T(val))), int>::value), "");
  assert(cuda::std::fpclassify(T(val)) == FP_NORMAL);
  assert(cuda::std::fpclassify(T(1.0)) == FP_NORMAL);
  assert(cuda::std::fpclassify(T(0.0)) == FP_ZERO);
  assert(cuda::std::fpclassify(T(-1.0)) == FP_NORMAL);
  assert(cuda::std::fpclassify(T(-0.0)) == FP_ZERO);
  // extended floating point types have issues here
  if (!cuda::std::__is_extended_floating_point<T>::value)
  {
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<T>::quiet_NaN()) == FP_NAN);
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<T>::infinity()) == FP_INFINITE);
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<T>::denorm_min()) == FP_SUBNORMAL);
  }
  else
  {
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<float>::quiet_NaN()) == FP_NAN);
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<float>::infinity()) == FP_INFINITE);
    // float subnormal turns to 0.0 for our half precision types
    assert(cuda::std::fpclassify(cuda::std::numeric_limits<float>::denorm_min()) == FP_ZERO);
  }
}

__host__ __device__ void test_fpclassify(float val)
{
  test_fpclassify<float>(val);
  test_fpclassify<double>(val);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_fpclassify<long double>(val);
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_fpclassify<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_fpclassify<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16

  assert(cuda::std::fpclassify(0u) == FP_ZERO);
  assert(cuda::std::fpclassify(cuda::std::numeric_limits<unsigned>::max()) == FP_NORMAL);
  assert(cuda::std::fpclassify(1) == FP_NORMAL);
  assert(cuda::std::fpclassify(-1) == FP_NORMAL);
  assert(cuda::std::fpclassify(cuda::std::numeric_limits<int>::max()) == FP_NORMAL);
  assert(cuda::std::fpclassify(cuda::std::numeric_limits<int>::min()) == FP_NORMAL);
}

template <class T>
__host__ __device__ void test_signbit(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::signbit((T) 0)), bool>::value), "");
  assert(cuda::std::signbit(T(val)) == false);
  assert(cuda::std::signbit(T(-1.0)) == true);
  assert(cuda::std::signbit(T(0.0)) == false);
}

__host__ __device__ void test_signbit(float val)
{
  test_signbit<float>(val);
  test_signbit<double>(val);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_signbit<long double>(val);
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_signbit<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_signbit<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16

  assert(cuda::std::signbit(0u) == false);
  assert(cuda::std::signbit(cuda::std::numeric_limits<unsigned>::max()) == false);
  assert(cuda::std::signbit(1) == false);
  assert(cuda::std::signbit(-1) == true);
  assert(cuda::std::signbit(cuda::std::numeric_limits<int>::max()) == false);
  assert(cuda::std::signbit(cuda::std::numeric_limits<int>::min()) == true);
}

template <class T>
__host__ __device__ void test_isfinite(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isfinite((T) 0)), bool>::value), "");
  assert(cuda::std::isfinite(T(val)) == true);
  assert(cuda::std::isfinite(T(-1.0f)) == true);
  assert(cuda::std::isfinite(T(1.0f)) == true);
  assert(cuda::std::isfinite(cuda::std::numeric_limits<T>::quiet_NaN()) == false);
  assert(cuda::std::isfinite(cuda::std::numeric_limits<T>::infinity()) == false);
  assert(cuda::std::isfinite(-cuda::std::numeric_limits<T>::infinity()) == false);
}

__host__ __device__ void test_isfinite(float val)
{
  test_isfinite<float>(val);
  test_isfinite<double>(val);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_isfinite<long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_isfinite<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_isfinite<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16

  assert(cuda::std::isfinite(0) == true);
  assert(cuda::std::isfinite(1) == true);
  assert(cuda::std::isfinite(-1) == true);
  assert(cuda::std::isfinite(cuda::std::numeric_limits<int>::max()) == true);
  assert(cuda::std::isfinite(cuda::std::numeric_limits<int>::min()) == true);
}

__host__ __device__ _CCCL_CONSTEXPR_ISFINITE bool test_constexpr_isfinite(float val)
{
  return cuda::std::isfinite(val);
}

template <class T>
__host__ __device__ void test_isnormal(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isnormal((T) 0)), bool>::value), "");
  assert(cuda::std::isnormal(T(val)) == true);
  assert(cuda::std::isnormal(T(-1.0f)) == true);
  assert(cuda::std::isnormal(T(1.0f)) == true);
  assert(cuda::std::isnormal(T(0.0f)) == false);
  assert(cuda::std::isnormal(cuda::std::numeric_limits<T>::quiet_NaN()) == false);
  assert(cuda::std::isnormal(cuda::std::numeric_limits<T>::infinity()) == false);
  assert(cuda::std::isnormal(-cuda::std::numeric_limits<T>::infinity()) == false);
}

__host__ __device__ void test_isnormal(float val)
{
  test_isnormal<float>(val);
  test_isnormal<double>(val);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_isnormal<long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_isnormal<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_isnormal<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16

  assert(cuda::std::isnormal(0) == false);
  assert(cuda::std::isnormal(1) == true);
  assert(cuda::std::isnormal(-1) == true);
  assert(cuda::std::isnormal(cuda::std::numeric_limits<int>::max()) == true);
  assert(cuda::std::isnormal(cuda::std::numeric_limits<int>::min()) == true);
}

__host__ __device__ void test_isgreater(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((double) 0, (long double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((long double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((long double) 0, (long double) 0)), bool>::value),
                "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isgreater(-1.0, 0.F) == false);
}

__host__ __device__ void test_isgreaterequal(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((double) 0, (long double) 0)), bool>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((long double) 0, (double) 0)), bool>::value),
                "");
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::isgreaterequal((long double) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (float) 0)), bool>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (double) 0)), bool>::value),
                "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isgreaterequal(-1.0, 0.F) == false);
}

__host__ __device__ void test_isinf(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isinf((float) 0)), bool>::value), "");

  typedef decltype(cuda::std::isinf((double) 0)) DoubleRetType;
  static_assert((cuda::std::is_same<DoubleRetType, bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isinf(0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isinf((long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isinf((__half) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::isinf((__nv_bfloat16) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isinf(-1.0) == false);
  assert(cuda::std::isinf(0) == false);
  assert(cuda::std::isinf(1) == false);
  assert(cuda::std::isinf(-1) == false);
  assert(cuda::std::isinf(cuda::std::numeric_limits<int>::max()) == false);
  assert(cuda::std::isinf(cuda::std::numeric_limits<int>::min()) == false);
}

__host__ __device__ _CCCL_CONSTEXPR_ISINF bool test_constexpr_isinf(float val)
{
  return cuda::std::isinf(val);
}

__host__ __device__ void test_isless(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((double) 0, (long double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((long double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((long double) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__nv_bfloat16) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isless((__nv_bfloat16) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isless(-1.0, 0.F) == true);
}

__host__ __device__ void test_islessequal(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((double) 0, (long double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((long double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((long double) 0, (long double) 0)), bool>::value),
                "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::islessequal(-1.0, 0.F) == true);
}

__host__ __device__ void test_islessgreater(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((double) 0, (long double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((long double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((long double) 0, (long double) 0)), bool>::value),
                "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (float) 0)), bool>::value),
                "");
  static_assert((cuda::std::is_same<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (double) 0)), bool>::value),
                "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::islessgreater(-1.0, 0.F) == true);
}

__host__ __device__ void test_isnan(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isnan((float) 0)), bool>::value), "");

  typedef decltype(cuda::std::isnan((double) 0)) DoubleRetType;
  static_assert((cuda::std::is_same<DoubleRetType, bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isnan(0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isnan((long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isnan((__half) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::isnan((__nv_bfloat16) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isnan(-1.0) == false);
  assert(cuda::std::isnan(0) == false);
  assert(cuda::std::isnan(1) == false);
  assert(cuda::std::isnan(-1) == false);
  assert(cuda::std::isnan(cuda::std::numeric_limits<int>::max()) == false);
  assert(cuda::std::isnan(cuda::std::numeric_limits<int>::min()) == false);
}

__host__ __device__ _CCCL_CONSTEXPR_ISNAN bool test_constexpr_isnan(float val)
{
  return cuda::std::isnan(val);
}

__host__ __device__ void test_isunordered(float val)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((float) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((float) 0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((float) 0, (long double) 0)), bool>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered(0, (double) 0)), bool>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((double) 0, (long double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((long double) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((long double) 0, (double) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((long double) 0, (long double) 0)), bool>::value),
                "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((__half) 0, (__half) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((__half) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((__half) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert(
    (cuda::std::is_same<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (float) 0)), bool>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (double) 0)), bool>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::isunordered(-1.0, 0.F) == false);
}

__host__ __device__ void test(float val)
{
  test_fpclassify(val);
  test_signbit(val);
  test_isfinite(val);
  test_isnormal(val);
  test_isgreater(val);
  test_isgreaterequal(val);
  test_isinf(val);
  test_isless(val);
  test_islessequal(val);
  test_islessgreater(val);
  test_isnan(val);
  test_isunordered(val);
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);

#if defined(_CCCL_BUILTIN_ISNAN)
  static_assert(!test_constexpr_isnan(1.0f), "");
#endif // _CCCL_BUILTIN_ISNAN

#if defined(_CCCL_BUILTIN_ISINF)
  static_assert(!test_constexpr_isinf(1.0f), "");
#endif // _CCCL_BUILTIN_ISINF

#if defined(_CCCL_BUILTIN_ISFINITE) || (defined(_CCCL_BUILTIN_ISINF) && defined(_CCCL_BUILTIN_ISNAN))
  static_assert(test_constexpr_isfinite(1.0f), "");
#endif // _CCCL_BUILTIN_ISFINITE|| (_CCCL_BUILTIN_ISINF && _CCCL_BUILTIN_ISNAN)

  return 0;
}
