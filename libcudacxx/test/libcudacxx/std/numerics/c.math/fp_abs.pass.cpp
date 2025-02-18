//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T, cuda::std::enable_if_t<cuda::is_floating_point_v<T>, int> = 0>
__host__ __device__ void test_fabs_abs(const T pos)
{
  ASSERT_SAME_TYPE(T, decltype(cuda::std::fabs(T{})));
  ASSERT_SAME_TYPE(T, decltype(cuda::std::abs(T{})));

  T result{};

  // 1. Test fabs on positive input
  result = cuda::std::fabs(pos);
  assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);

  // 2. Test abs on positive input
  result = cuda::std::abs(pos);
  assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);

  // 3. Test fabsf and fabsl on positive input
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    result = cuda::std::fabsf(pos);
    assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);
  }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if constexpr (cuda::std::is_same_v<T, long double>)
  {
    result = cuda::std::fabsl(pos);
    assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);
  }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T neg = cuda::std::copysign(pos, cuda::std::numeric_limits<T>::lowest());

    // 4. Test fabs on negative input
    result = cuda::std::fabs(neg);
    assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);

    // 5. Test abs on negative input
    result = cuda::std::abs(neg);
    assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);

    // 6. Test fabsf and fabsl on negative input
    if constexpr (cuda::std::is_same_v<T, float>)
    {
      result = cuda::std::fabsf(neg);
      assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);
    }
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
    else if constexpr (cuda::std::is_same_v<T, long double>)
    {
      result = cuda::std::fabsl(neg);
      assert(cuda::std::memcmp(&result, &pos, sizeof(T)) == 0);
    }
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  }
}

template <class T, cuda::std::enable_if_t<cuda::std::is_integral_v<T>, int> = 0>
__host__ __device__ void test_fabs_abs(const T pos)
{
  ASSERT_SAME_TYPE(double, decltype(cuda::std::fabs(T{})));

  const double pos_ref = cuda::std::fabs(static_cast<double>(pos));
  double result{};

  // 1. Test fabs on positive input
  result = cuda::std::fabs(pos);
  assert(cuda::std::memcmp(&result, &pos_ref, sizeof(double)) == 0);

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T neg          = -pos;
    const double neg_ref = cuda::std::fabs(static_cast<double>(neg));

    // 2. Test fabs on negative input
    result = cuda::std::fabs(neg);
    assert(cuda::std::memcmp(&result, &neg_ref, sizeof(double)) == 0);
  }
}

template <class T>
__host__ __device__ void test_type()
{
  // __nv_fp8_e8m0 cannot represent 0
#if _CCCL_HAS_NVFP8_E8M0()
  if constexpr (!cuda::std::is_same_v<T, __nv_fp8_e8m0>)
#endif // _CCCL_HAS_NVFP8_E8M0
  {
    test_fabs_abs(T{});
  }
  test_fabs_abs(static_cast<T>(1));
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

__host__ __device__ bool test()
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
  // static_assert(test(0.f), "");
  return 0;
}
