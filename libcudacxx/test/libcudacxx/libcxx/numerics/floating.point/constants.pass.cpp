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
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_fp_storage(cuda::std::__fp_storage_of_t<T> expected_one)
{
  constexpr auto fmt = cuda::std::__fp_format_of_v<T>;

  // __fp_has_inf_v must match numeric_limits::has_infinity
  static_assert(cuda::std::__fp_has_inf_v<fmt> == cuda::std::numeric_limits<T>::has_infinity);

  // test __fp_inf value to match numeric_limits::infinity()
  if constexpr (cuda::std::__fp_has_inf_v<fmt>)
  {
    const auto val = cuda::std::__fp_inf<T>();
    const auto ref = cuda::std::numeric_limits<T>::infinity();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }

  // __fp_has_nan_v must match numeric_limits::has_quiet_NaN
  static_assert(cuda::std::__fp_has_nan_v<fmt> == cuda::std::numeric_limits<T>::has_quiet_NaN);

  // test __fp_nan value
  if constexpr (cuda::std::__fp_has_nan_v<fmt>)
  {
    assert(cuda::std::isnan(cuda::std::__fp_nan<T>()));
  }

  // __fp_has_nans_v must match numeric_limits::has_signaling_NaN
  static_assert(cuda::std::__fp_has_nans_v<fmt> == cuda::std::numeric_limits<T>::has_signaling_NaN);

  // test __fp_nans value
  if constexpr (cuda::std::__fp_has_nans_v<fmt>)
  {
    assert(cuda::std::isnan(cuda::std::__fp_nans<T>()));
  }

  // test __fp_max value to match numeric_limits::max()
  {
    const auto val = cuda::std::__fp_max<T>();
    const auto ref = cuda::std::numeric_limits<T>::max();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }

  // test __fp_min value to match numeric_limits::min()
  {
    const auto val = cuda::std::__fp_min<T>();
    const auto ref = cuda::std::numeric_limits<T>::min();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }

  // test __fp_lowest value to match numeric_limits::lowest()
  {
    const auto val = cuda::std::__fp_lowest<T>();
    const auto ref = cuda::std::numeric_limits<T>::lowest();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }

  // test __fp_zero to be all zeros
#if _CCCL_HAS_NVFP8_E8M0()
  if constexpr (!cuda::std::is_same_v<T, __nv_fp8_e8m0>)
#endif // _CCCL_HAS_NVFP8_E8M0()
  {
    const auto val = cuda::std::__fp_zero<T>();
    const auto ref = cuda::std::__fp_storage_of_t<T>(0);
    assert(cuda::std::__fp_zero<cuda::std::__fp_format_of_v<T>>() == cuda::std::__fp_storage_of_t<T>(0));
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
    assert(!cuda::std::signbit(val));
  }

  // test __fp_one for standard types only
  {
    const auto val = cuda::std::__fp_one<T>();
    assert(cuda::std::__fp_one<cuda::std::__fp_format_of_v<T>>() == expected_one);
    assert(cuda::std::__fp_get_storage(val) == expected_one);
  }
}

int main(int, char**)
{
  test_fp_storage<float>(0x3f800000);
  test_fp_storage<double>(0x3ff0000000000000);
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_storage<long double>(0x3fff8000000000000000);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_storage<__half>(0x3c00);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_storage<__nv_bfloat16>(0x3f80);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_storage<__nv_fp8_e4m3>(0x38);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_storage<__nv_fp8_e5m2>(0x3c);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_storage<__nv_fp8_e8m0>(0x7f);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_storage<__nv_fp6_e2m3>(0x8);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_storage<__nv_fp6_e3m2>(0xc);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_storage<__nv_fp4_e2m1>(0x2);
#endif // _CCCL_HAS_NVFP4_E2M1()

  return 0;
}
