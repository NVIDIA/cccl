//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_fp_get_exp(T val, int expected)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::__fp_get_exp(cuda::std::declval<T>())), int>);
  static_assert(noexcept(cuda::std::__fp_get_exp(cuda::std::declval<T>())));
  assert(cuda::std::__fp_get_exp(val) == expected);
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_fp_get_exp(T val)
{
  constexpr auto fmt = cuda::std::__fp_format_of_v<T>;

  test_fp_get_exp(val, 0);
  if constexpr (cuda::std::__fp_exp_nbits_v<fmt> > 3)
  {
    test_fp_get_exp(T{32}, 5);
  }

  // zero has ... all zeros
  if constexpr (cuda::std::__fp_is_signed_v<fmt>)
  {
    test_fp_get_exp(T{}, cuda::std::__fp_exp_min_v<fmt> - 1);
  }
  else
  { // Without sign 0 is equivalent to __fp_exp_min_v
    test_fp_get_exp(T{}, cuda::std::__fp_exp_min_v<fmt>);
  }

  // min has all zeroes, max and lowest have max
  test_fp_get_exp(cuda::std::numeric_limits<T>::min(), cuda::std::__fp_exp_min_v<fmt>);
  test_fp_get_exp(cuda::std::numeric_limits<T>::max(), cuda::std::__fp_exp_max_v<fmt>);
  if constexpr (cuda::std::__fp_is_signed_v<fmt>)
  {
    test_fp_get_exp(cuda::std::numeric_limits<T>::lowest(), cuda::std::__fp_exp_max_v<fmt>);
  }
  else
  { // Without sign lowest is equivalent to min
    test_fp_get_exp(cuda::std::numeric_limits<T>::lowest(), cuda::std::__fp_exp_min_v<fmt>);
  }

  // Denormals have all zeros, so its one less than __fp_exp_min_v
  if constexpr (cuda::std::__fp_has_denorm_v<fmt>)
  {
    test_fp_get_exp(cuda::std::numeric_limits<T>::denorm_min(), cuda::std::__fp_exp_min_v<fmt> - 1);
  }

  // infinity and NaN have full zeros, so one more than __fp_exp_max_v
  if constexpr (cuda::std::__fp_has_inf_v<fmt>)
  {
    test_fp_get_exp(cuda::std::numeric_limits<T>::infinity(), cuda::std::__fp_exp_max_v<fmt> + 1);
  }
  if constexpr (cuda::std::__fp_has_nan_v<fmt>)
  {
    if constexpr (cuda::std::is_same_v<T, __nv_fp8_e4m3>)
    {
      // __nv_fp8_e4m3 has only 2 NaNs so more of exponent are valid
      test_fp_get_exp(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::__fp_exp_max_v<fmt>);
    }
    else
    {
      test_fp_get_exp(cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::__fp_exp_max_v<fmt> + 1);
    }
  }
  if constexpr (cuda::std::__fp_has_nans_v<fmt>)
  {
    test_fp_get_exp(cuda::std::numeric_limits<T>::signaling_NaN(), cuda::std::__fp_exp_max_v<fmt> + 1);
  }
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test(T val = T{1.0f})
{
  test_fp_get_exp<T>(val);
}

__host__ __device__ bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  return true;
}

__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST bool test_constexpr(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

int main(int, char**)
{
  volatile float value = 1.0f;
  test(value);

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test_constexpr(1.0f));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

  return 0;
}
