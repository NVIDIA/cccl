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
    static_assert(cuda::std::is_same_v<T, decltype(cuda::std::copysignf(T{}, T{}))>);
    const auto resultf = cuda::std::copysignf(mag, sign);
    assert(cuda::std::signbit(resultf) == expected);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    static_assert(cuda::std::is_same_v<T, decltype(cuda::std::copysignl(T{}, T{}))>);
    const auto resultl = cuda::std::copysignl(mag, sign);
    assert(cuda::std::signbit(resultl) == expected);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
}

template <class T>
__host__ __device__ constexpr void test_copysign(const T pos)
{
  using Result = cuda::std::conditional_t<cuda::std::is_integral_v<T>, double, T>;

  static_assert(cuda::std::is_same_v<Result, decltype(cuda::std::copysign(T{}, T{}))>);

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
__host__ __device__ constexpr void test_type(float val)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
#if TEST_CUDA_COMPILER(CLANG)
    if constexpr (sizeof(T) < 16) // clang fails the conversion from float to 128 bit integers
#endif // TEST_CUDA_COMPILER(CLANG)
    {
      test_copysign(static_cast<T>(val));
    }
  }
  else
  {
    test_copysign(cuda::std::__fp_cast<T>(val));
  }

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

__host__ __device__ constexpr bool test(float val)
{
  test_type<float>(val);
  test_type<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_type<__half>(val);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>(val);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>(val);
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>(val);
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>(val);
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>(val);
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>(val);
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>(val);
#endif // _CCCL_HAS_NVFP4_E2M1

  test_type<signed char>(val);
  test_type<unsigned char>(val);
  test_type<signed short>(val);
  test_type<unsigned short>(val);
  test_type<signed int>(val);
  test_type<unsigned int>(val);
  test_type<signed long>(val);
  test_type<unsigned long>(val);
  test_type<signed long long>(val);
  test_type<unsigned long long>(val);
#if _CCCL_HAS_INT128()
  test_type<__int128_t>(val);
  test_type<__uint128_t>(val);
#endif // _CCCL_HAS_INT128()

  return true;
}

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
__host__ __device__ constexpr bool test_constexpr(float val)
{
  test_type<float>(val);
  test_type<double>(val);
#  if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>(val);
#  endif // _CCCL_HAS_LONG_DOUBLE()

  test_type<signed char>(val);
  test_type<unsigned char>(val);
  test_type<signed short>(val);
  test_type<unsigned short>(val);
  test_type<signed int>(val);
  test_type<unsigned int>(val);
  test_type<signed long>(val);
  test_type<unsigned long>(val);
  test_type<signed long long>(val);
  test_type<unsigned long long>(val);
#  if _CCCL_HAS_INT128()
  test_type<__int128_t>(val);
  test_type<__uint128_t>(val);
#  endif // _CCCL_HAS_INT128()

  return true;
}
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test_constexpr(1.0f));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

  return 0;
}
