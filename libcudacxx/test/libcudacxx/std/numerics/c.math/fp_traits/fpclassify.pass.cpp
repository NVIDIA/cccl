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

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_fpclassify(T val, int expected)
{
  assert(cuda::std::fpclassify(val) == expected);

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    T neg{};

    if constexpr (cuda::std::numeric_limits<T>::is_integer)
    {
      // handle integer overflow when negating the minimum value
      neg = (val == cuda::std::numeric_limits<T>::min()) ? cuda::std::numeric_limits<T>::max() : -val;
    }
    else if constexpr (cuda::std::is_floating_point_v<T>)
    {
      neg = -val;
    }
    else // nvfp types
    {
      neg = cuda::std::copysign(val, cuda::std::numeric_limits<T>::lowest());
    }

    assert(cuda::std::fpclassify(neg) == expected);
  }
}

template <class T, cuda::std::enable_if_t<cuda::is_floating_point_v<T>, int> = 0>
__host__ __device__ constexpr void test_type()
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::fpclassify(T{})));

  // __nv_fp8_e8m0 cannot represent 0
#if _CCCL_HAS_NVFP8_E8M0()
  if constexpr (!cuda::std::is_same_v<T, __nv_fp8_e8m0>)
#endif // _CCCL_HAS_NVFP8_E8M0
  {
    test_fpclassify(T{}, FP_ZERO);
  }
  test_fpclassify(cuda::std::numeric_limits<T>::min(), FP_NORMAL);
  test_fpclassify(cuda::std::numeric_limits<T>::max(), FP_NORMAL);

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    test_fpclassify(cuda::std::numeric_limits<T>::infinity(), FP_INFINITE);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    test_fpclassify(cuda::std::numeric_limits<T>::quiet_NaN(), FP_NAN);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_signaling_NaN)
  {
    test_fpclassify(cuda::std::numeric_limits<T>::signaling_NaN(), FP_NAN);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_denorm)
  {
    // fixme: behaviour of subnormal values depends on the FP mode, may result in FP_ZERO
    // test_fpclassify(cuda::std::numeric_limits<T>::denorm_min(), FP_SUBNORMAL);
  }
}

template <class T, cuda::std::enable_if_t<cuda::std::is_integral_v<T>, int> = 0>
__host__ __device__ constexpr void test_type()
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::fpclassify(T{})));

  test_fpclassify(T(1), FP_NORMAL);
  test_fpclassify(T(0), FP_ZERO);
  test_fpclassify(cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::is_signed ? FP_NORMAL : FP_ZERO);
  test_fpclassify(cuda::std::numeric_limits<T>::max(), FP_NORMAL);
}

__host__ __device__ constexpr bool test()
{
  test_type<float>();
  test_type<double>();
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_type<long double>();
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
#if defined(_LIBCUDACXX_HAS_NVFP16)
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#if defined(_LIBCUDACXX_HAS_NVBF16)
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16
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

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
