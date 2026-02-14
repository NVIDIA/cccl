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

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

template <class T>
__host__ __device__ constexpr void test_fp_neg()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::__fp_neg(T{})), T>);
  static_assert(noexcept(cuda::std::__fp_neg(T{})));

  // todo: implement test once __fp_cast is implemented
}

__host__ __device__ constexpr bool test()
{
  // standard floating point types
  test_fp_neg<float>();
  test_fp_neg<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_neg<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  // extended nvidia floating point types
#if _CCCL_HAS_NVFP16()
  test_fp_neg<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_neg<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_neg<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_neg<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_neg<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_neg<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_neg<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_neg<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  // extended compiler floating point types
#if _CCCL_HAS_FLOAT128()
  test_fp_neg<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  // extended cccl floating point types
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary16>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary32>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary64>>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__binary128>>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__bfloat16>>();
#if _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp80_x86>>();
#endif // _CCCL_HAS_INT128()
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e4m3>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e5m2>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e8m0>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e2m3>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e3m2>>();
  test_fp_neg<cuda::std::__cccl_fp<cuda::std::__fp_format::__fp4_nv_e2m1>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
