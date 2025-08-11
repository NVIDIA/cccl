//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_fp_format(cuda::std::__fp_format expected)
{
  assert(cuda::std::__fp_format_of_v<T> == expected);
}

__host__ __device__ constexpr bool test()
{
  test_fp_format<float>(cuda::std::__fp_format::__binary32);
  test_fp_format<double>(cuda::std::__fp_format::__binary64);
#if _CCCL_HAS_LONG_DOUBLE()
  if (cuda::std::numeric_limits<long double>::min_exponent == cuda::std::numeric_limits<double>::min_exponent
      && cuda::std::numeric_limits<long double>::max_exponent == cuda::std::numeric_limits<double>::max_exponent
      && cuda::std::numeric_limits<long double>::digits == cuda::std::numeric_limits<double>::digits)
  {
    test_fp_format<long double>(cuda::std::__fp_format::__binary64);
  }
  else if (cuda::std::numeric_limits<long double>::min_exponent == -16381
           && cuda::std::numeric_limits<long double>::max_exponent == 16384
           && cuda::std::numeric_limits<long double>::digits == 64 && sizeof(long double) == 16)
  {
    test_fp_format<long double>(cuda::std::__fp_format::__fp80_x86);
  }
  else if (cuda::std::numeric_limits<long double>::min_exponent == -16381
           && cuda::std::numeric_limits<long double>::max_exponent == 16384
           && cuda::std::numeric_limits<long double>::digits == 113)
  {
    test_fp_format<long double>(cuda::std::__fp_format::__binary128);
  }
  else
  {
    test_fp_format<long double>(cuda::std::__fp_format::__invalid);
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_format<__half>(cuda::std::__fp_format::__binary16);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_format<__nv_bfloat16>(cuda::std::__fp_format::__bfloat16);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_format<__nv_fp8_e4m3>(cuda::std::__fp_format::__fp8_nv_e4m3);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_format<__nv_fp8_e5m2>(cuda::std::__fp_format::__fp8_nv_e5m2);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_format<__nv_fp8_e8m0>(cuda::std::__fp_format::__fp8_nv_e8m0);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_format<__nv_fp6_e2m3>(cuda::std::__fp_format::__fp6_nv_e2m3);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_format<__nv_fp6_e3m2>(cuda::std::__fp_format::__fp6_nv_e3m2);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_format<__nv_fp4_e2m1>(cuda::std::__fp_format::__fp4_nv_e2m1);
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_fp_format<__float128>(cuda::std::__fp_format::__binary128);
#endif // _CCCL_HAS_FLOAT128()

  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary16>>(cuda::std::__fp_format::__binary16);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary32>>(cuda::std::__fp_format::__binary32);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary64>>(cuda::std::__fp_format::__binary64);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary128>>(cuda::std::__fp_format::__binary128);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp80_x86>>(cuda::std::__fp_format::__fp80_x86);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__bfloat16>>(cuda::std::__fp_format::__bfloat16);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e4m3>>(cuda::std::__fp_format::__fp8_nv_e4m3);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e5m2>>(cuda::std::__fp_format::__fp8_nv_e5m2);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e8m0>>(cuda::std::__fp_format::__fp8_nv_e8m0);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e2m3>>(cuda::std::__fp_format::__fp6_nv_e2m3);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e3m2>>(cuda::std::__fp_format::__fp6_nv_e3m2);
  test_fp_format<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp4_nv_e2m1>>(cuda::std::__fp_format::__fp4_nv_e2m1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
