//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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
__host__ __device__ constexpr void test_eq(const T tested, const T expected)
{
  if (cuda::std::isnan(expected))
  {
    assert(cuda::std::isnan(tested));
    assert(cuda::std::signbit(tested) == false);
    return;
  }

  // extended floating point types don't support constexpr operator==
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    assert(tested == expected);
  }
  else
  {
    assert(cuda::std::__fp_get_storage(tested) == cuda::std::__fp_get_storage(expected));
  }
}

template <class T, cuda::std::enable_if_t<cuda::is_floating_point_v<T>, int> = 0>
__host__ __device__ void constexpr test_fabs_abs(const T pos)
{
  ASSERT_SAME_TYPE(T, decltype(cuda::std::fabs(T{})));
  ASSERT_SAME_TYPE(T, decltype(cuda::std::abs(T{})));

  // 1. Test fabs on positive input
  test_eq(cuda::std::fabs(pos), pos);

  // 2. Test abs on positive input
  test_eq(cuda::std::abs(pos), pos);

  // 3. Test fabsf and fabsl on positive input
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    test_eq(cuda::std::fabsf(pos), pos);
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    test_eq(cuda::std::fabsl(pos), pos);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T neg = cuda::std::copysign(pos, cuda::std::numeric_limits<T>::lowest());

    // 4. Test fabs on negative input
    test_eq(cuda::std::fabs(neg), pos);

    // 5. Test abs on negative input
    test_eq(cuda::std::abs(neg), pos);

    // 6. Test fabsf and fabsl on negative input
    if constexpr (cuda::std::is_same_v<T, float>)
    {
      test_eq(cuda::std::fabsf(neg), pos);
    }
#if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (cuda::std::is_same_v<T, long double>)
    {
      test_eq(cuda::std::fabsl(neg), pos);
    }
#endif // _CCCL_HAS_LONG_DOUBLE()
  }
}

template <class T, cuda::std::enable_if_t<cuda::std::is_integral_v<T>, int> = 0>
__host__ __device__ void constexpr test_fabs_abs(const T pos)
{
  ASSERT_SAME_TYPE(double, decltype(cuda::std::fabs(T{})));

  const double pos_ref = cuda::std::fabs(static_cast<double>(pos));

  // 1. Test fabs on positive input
  test_eq(cuda::std::fabs(pos), pos_ref);

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T neg          = (pos == cuda::std::numeric_limits<T>::min()) ? cuda::std::numeric_limits<T>::max() : -pos;
    const double neg_ref = cuda::std::fabs(static_cast<double>(neg));

    // 2. Test fabs on negative input
    test_eq(cuda::std::fabs(neg), neg_ref);
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
    test_fabs_abs(T{});
  }
  test_fabs_abs(cuda::std::numeric_limits<T>::min());
  test_fabs_abs(cuda::std::numeric_limits<T>::max());

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    test_fabs_abs(cuda::std::numeric_limits<T>::infinity());
  }

  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    test_fabs_abs(cuda::std::numeric_limits<T>::quiet_NaN());
  }

  if constexpr (cuda::std::numeric_limits<T>::has_signaling_NaN)
  {
    test_fabs_abs(cuda::std::numeric_limits<T>::signaling_NaN());
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

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test_constexpr());
  return 0;
}
