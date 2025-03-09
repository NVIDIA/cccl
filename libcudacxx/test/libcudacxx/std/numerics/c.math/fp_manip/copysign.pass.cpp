//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_copysign(const T mag, const T sign, bool expected)
{
  const auto result = cuda::std::copysign(mag, sign);
  assert(cuda::std::signbit(result) == expected);

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    const auto result = cuda::std::copysignf(mag, sign);
    assert(cuda::std::signbit(result) == expected);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    const auto result = cuda::std::copysignl(mag, sign);
    assert(cuda::std::signbit(result) == expected);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <class T>
__host__ __device__ constexpr void test_copysign(const T pos)
{
  using Result = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;

  ASSERT_SAME_TYPE(Result, decltype(cuda::std::copysign(T{}, T{})));

  // 1. positive -> positive
  test_copysign<T>(pos, cuda::std::numeric_limits<T>::max(), false);

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    T neg{};
    if constexpr (cuda::std::is_integral_v<T>)
    {
      neg = -pos;
    }
    else
    {
      neg = cuda::std::copysign(pos, cuda::std::numeric_limits<T>::lowest());
    }

    // 2. positive -> negative
    test_copysign<T>(pos, cuda::std::numeric_limits<T>::lowest(), true);

    // 3. negative -> negative
    test_copysign<T>(neg, cuda::std::numeric_limits<T>::lowest(), true);

    // 4. negative -> positive
    test_copysign<T>(neg, cuda::std::numeric_limits<T>::max(), false);
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  // __nv_fp8_e8m0 cannot represent 0
#if _CCCL_HAS_NVFP8_E8M0()
  if constexpr (!cuda::std::is_same_v<T, __nv_fp8_e8m0>)
#endif // _CCCL_HAS_NVFP8_E8M0
  {
    test_copysign(T{});
  }
  if constexpr (!cuda::std::numeric_limits<T>::is_integer)
  {
    test_copysign(cuda::std::numeric_limits<T>::min());
  }
  test_copysign(cuda::std::numeric_limits<T>::max());

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    test_copysign(cuda::std::numeric_limits<T>::infinity());
  }

  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    test_copysign(cuda::std::numeric_limits<T>::quiet_NaN());
  }

  if constexpr (cuda::std::numeric_limits<T>::has_signaling_NaN)
  {
    test_copysign(cuda::std::numeric_limits<T>::signaling_NaN());
  }

  if constexpr (cuda::std::numeric_limits<T>::has_denorm)
  {
    test_copysign(cuda::std::numeric_limits<T>::denorm_min());
  }
}

__host__ __device__ constexpr bool test()
{
  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  test_type<signed char>();
  test_type<unsigned char>();
  test_type<signed short>();
  test_type<unsigned short>();
  test_type<signed int>();
  test_type<unsigned int>();
  test_type<signed long>();
  test_type<unsigned long>();
  test_type<signed long long>();
  test_type<unsigned long long>();
#if !defined(TEST_HAS_NO_INT128_T)
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // !TEST_HAS_NO_INT128_T

  return true;
}

__host__ __device__ constexpr bool test_constexpr()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  // we cannot test integral types, because signbit(double) is not constexpr

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test_constexpr());
  return 0;
}
