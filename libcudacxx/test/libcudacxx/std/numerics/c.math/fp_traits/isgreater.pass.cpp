//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

// <cmath>

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

// numeric_limits::has_denorm has been deprecated since C++23
#if _CCCL_STD_VER >= 2023
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
#endif // _CCCL_STD_VER >= 2023

template <class T, class U>
TEST_FUNC constexpr void test_isgreater(T x, U y, bool expected)
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::isgreater(x, y))>);
  static_assert(noexcept(cuda::std::isgreater(x, y)));

  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(x);
    DoNotOptimize(y);
  }
  assert(cuda::std::isgreater(x, y) == expected);
}

template <class T>
TEST_FUNC constexpr void test_ordered(T low, T high)
{
  test_isgreater(low, high, false);
  test_isgreater(high, low, true);
  test_isgreater(low, low, false);
  test_isgreater(high, high, false);
}

template <class T>
TEST_FUNC constexpr void test_nan(T nan)
{
  T max = cuda::std::numeric_limits<T>::max();
  test_isgreater(nan, max, false);
  test_isgreater(max, nan, false);
  test_isgreater(nan, nan, false);
  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    T negative_nan = cuda::std::copysign(nan, cuda::std::numeric_limits<T>::lowest());
    test_isgreater(negative_nan, max, false);
    test_isgreater(max, negative_nan, false);
  }
}

template <class T>
TEST_FUNC constexpr void test_type()
{
  using Limits = cuda::std::numeric_limits<T>;
  T min        = Limits::min();
  T max        = Limits::max();
  test_ordered(min, max);

  if constexpr (Limits::is_signed)
  {
    T zero{};
    T negative_zero = cuda::std::copysign(zero, Limits::lowest());
    T negative_min  = cuda::std::copysign(min, Limits::lowest());
    test_ordered(Limits::lowest(), negative_min);
    test_ordered(negative_min, min);
    test_ordered(negative_min, zero);
    test_ordered(zero, min);
    test_isgreater(zero, negative_zero, false);
    test_isgreater(negative_zero, zero, false);
    test_isgreater(negative_zero, negative_zero, false);
    // A nonzero denorm_min is smaller than min; formats without subnormals return zero.
    T denorm = Limits::denorm_min();
    if constexpr (Limits::has_denorm == cuda::std::denorm_absent)
    {
      test_isgreater(denorm, zero, false);
    }
    else
    {
      test_ordered(zero, denorm);
      test_ordered(denorm, min);
      test_ordered(cuda::std::copysign(denorm, Limits::lowest()), negative_zero);
    }
  }

  if constexpr (Limits::has_infinity)
  {
    test_ordered(max, Limits::infinity());
    test_ordered(cuda::std::copysign(Limits::infinity(), Limits::lowest()), Limits::lowest());
  }
  if constexpr (Limits::has_quiet_NaN)
  {
    test_nan(Limits::quiet_NaN());
  }
  if constexpr (Limits::has_signaling_NaN)
  {
    test_nan(Limits::signaling_NaN());
  }
}

TEST_FUNC constexpr bool test()
{
  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_type<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_isgreater(-1, 0.0f, false);
  test_isgreater(0.0f, -1, true);
  test_isgreater(-0.0, 0, false);
  test_isgreater(0, -0.0, false);
  test_isgreater(1.0f, 2.0, false);
  test_isgreater(2.0, 1.0f, true);
  test_isgreater(1, 1u, false);
  test_isgreater(-1, 0, false);
  test_isgreater(0u, 1u, false);
  test_isgreater(false, true, false);
  // Integral arguments are promoted to double, including arguments of the same type.
  test_isgreater(9007199254740992LL, 9007199254740993LL, false);
  test_isgreater(cuda::std::numeric_limits<float>::quiet_NaN(), 1.0, false);
  test_isgreater(1.0f, cuda::std::numeric_limits<double>::quiet_NaN(), false);
#if _CCCL_HAS_LONG_DOUBLE()
  test_isgreater(1.0L, 2.0, false);
  test_isgreater(2.0f, 1.0L, true);
#endif // _CCCL_HAS_LONG_DOUBLE()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
