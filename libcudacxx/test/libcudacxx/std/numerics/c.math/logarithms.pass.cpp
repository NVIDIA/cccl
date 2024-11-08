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

__host__ __device__ void test_log()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::log((float) 0)), float>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((bool) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((unsigned short) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((unsigned long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log((double) 0)), double>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::logf(0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::log((__half) 0)), __half>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::log((__nv_bfloat16) 0)), __nv_bfloat16>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::log(1) == 0);
}

__host__ __device__ void test_log10()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((float) 0)), float>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((bool) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((unsigned short) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((unsigned long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((double) 0)), double>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::log10f(0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log10l((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((__half) 0)), __half>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::log10((__nv_bfloat16) 0)), __nv_bfloat16>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::log10(1) == 0);
}

__host__ __device__ void test_ilogb()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((float) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((bool) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((unsigned short) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((int) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((unsigned int) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((long) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((unsigned long) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((long long) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((unsigned long long) 0)), int>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((double) 0)), int>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((long double) 0)), int>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogbf(0)), int>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogbl(0)), int>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((__half) 0)), int>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::ilogb((__nv_bfloat16) 0)), int>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::ilogb(1) == 0);
}

__host__ __device__ void test_log1p()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((float) 0)), float>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((bool) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((unsigned short) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((unsigned long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((double) 0)), double>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::log1pf(0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log1pl(0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((__half) 0)), __half>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::log1p((__nv_bfloat16) 0)), __nv_bfloat16>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::log1p(0) == 0);
}

__host__ __device__ void test_log2()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((float) 0)), float>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((bool) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((unsigned short) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((unsigned long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((double) 0)), double>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::log2f(0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::log2l(0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((__half) 0)), __half>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::log2((__nv_bfloat16) 0)), __nv_bfloat16>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::log2(1) == 0);
}

__host__ __device__ void test_logb()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((float) 0)), float>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((bool) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((unsigned short) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((unsigned int) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((unsigned long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((unsigned long long) 0)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((double) 0)), double>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((long double) 0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  static_assert((cuda::std::is_same<decltype(cuda::std::logbf(0)), float>::value), "");
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  static_assert((cuda::std::is_same<decltype(cuda::std::logbl(0)), long double>::value), "");
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#ifdef _LIBCUDACXX_HAS_NVFP16
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((__half) 0)), __half>::value), "");
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  static_assert((cuda::std::is_same<decltype(cuda::std::logb((__nv_bfloat16) 0)), __nv_bfloat16>::value), "");
#endif // _LIBCUDACXX_HAS_NVBF16
  assert(cuda::std::logb(1) == 0);
}

__host__ __device__ void test()
{
  test_log();
  test_log10();
  test_ilogb();
  test_log1p();
  test_log2();
  test_logb();
}

__global__ void test_global_kernel()
{
  test();
}

int main(int, char**)
{
  test();
  return 0;
}
