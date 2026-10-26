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

TEST_FUNC void test_isgreater(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isgreater(-1.0, 0.F) == false);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::isgreater((__float128) 1, 0.F) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_isgreaterequal(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert(
    (cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isgreaterequal(-1.0, 0.F) == false);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::isgreaterequal((__float128) 0, 0.F) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_isless(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isless(-1.0, 0.F) == true);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::isless((__float128) -1, 0.F) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_islessequal(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::islessequal(-1.0, 0.F) == true);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::islessequal((__float128) 0, 0.F) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_islessgreater(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert(
    (cuda::std::is_same_v<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::islessgreater(-1.0, 0.F) == true);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::islessgreater((__float128) -1, 0.F) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_isunordered(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (__nv_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__nv_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isunordered(-1.0, 0.F) == false);
#if _CCCL_HAS_FLOAT128()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__float128) 0, (__float128) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__float128) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__float128) 0, (double) 0)), bool>) );
  assert(cuda::std::isunordered((__float128) -1, 0.F) == false);
  assert(cuda::std::isunordered(cuda::std::numeric_limits<__float128>::quiet_NaN(), (__float128) 0) == true);
#endif // _CCCL_HAS_FLOAT128()
}

TEST_FUNC void test_fp64_comparison(
  double x, double y, bool less, bool less_equal, bool greater, bool greater_equal, bool less_greater)
{
  assert(cuda::std::isless(x, y) == less);
  assert(cuda::std::islessequal(x, y) == less_equal);
  assert(cuda::std::isgreater(x, y) == greater);
  assert(cuda::std::isgreaterequal(x, y) == greater_equal);
  assert(cuda::std::islessgreater(x, y) == less_greater);
}

// double has an integer implementation on slow fp64 targets
// check the cases where the integer mapping can go wrong
TEST_FUNC void test_fp64_edge_cases(float val)
{
  // values are derived from val at runtime
  double one   = val;
  double two   = one + one;
  double zero  = one - one;
  double nzero = -zero;
  double inf   = cuda::std::numeric_limits<double>::infinity() * one;
  double max   = cuda::std::numeric_limits<double>::max() * one;
  double dmin  = cuda::std::numeric_limits<double>::denorm_min() * one;
  double nan   = cuda::std::numeric_limits<double>::quiet_NaN();
  double nnan  = cuda::std::copysign(nan, -one);

  //                   x      y      <      <=     >      >=     <>
  // signed zeros compare equal
  test_fp64_comparison(zero, nzero, false, true, false, true, false);
  test_fp64_comparison(nzero, zero, false, true, false, true, false);

  // negative values are ordered in reverse to their bit patterns
  test_fp64_comparison(-one, -two, false, false, true, true, true);
  test_fp64_comparison(-two, -one, true, true, false, false, true);
  test_fp64_comparison(-one, one, true, true, false, false, true);
  test_fp64_comparison(one, -one, false, false, true, true, true);

  // equal values
  test_fp64_comparison(one, one, false, true, false, true, false);
  test_fp64_comparison(-one, -one, false, true, false, true, false);

  // infinities and limits
  test_fp64_comparison(-inf, inf, true, true, false, false, true);
  test_fp64_comparison(inf, inf, false, true, false, true, false);
  test_fp64_comparison(-inf, -max, true, true, false, false, true);
  test_fp64_comparison(max, inf, true, true, false, false, true);

  // denormals
  test_fp64_comparison(dmin, zero, false, false, true, true, true);
  test_fp64_comparison(-dmin, zero, true, true, false, false, true);
  test_fp64_comparison(-dmin, nzero, true, true, false, false, true);
  test_fp64_comparison(-dmin, dmin, true, true, false, false, true);

  // NaNs are unordered, including negative NaNs whose bit patterns would map below -inf
  test_fp64_comparison(nan, one, false, false, false, false, false);
  test_fp64_comparison(one, nan, false, false, false, false, false);
  test_fp64_comparison(nan, nan, false, false, false, false, false);
  test_fp64_comparison(nan, inf, false, false, false, false, false);
  test_fp64_comparison(nnan, -one, false, false, false, false, false);
  test_fp64_comparison(-inf, nnan, false, false, false, false, false);

  // mixed arguments are promoted to double
  int int_one = static_cast<int>(val);
  assert(cuda::std::isless(-int_one, -0.5 * one) == true);
  assert(cuda::std::isgreater(int_one - int_one, nzero) == false);
  assert(cuda::std::islessequal(int_one - int_one, nzero) == true);
  assert(cuda::std::islessgreater(-2 * int_one, -one) == true);
  assert(cuda::std::isless(-val, -0.5 * one) == true);
}

TEST_FUNC void test(float val)
{
  test_isgreater(val);
  test_isgreaterequal(val);
  test_isless(val);
  test_islessequal(val);
  test_islessgreater(val);
  test_isunordered(val);
  test_fp64_edge_cases(val);
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);

  return 0;
}
