//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license minormation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>

#include "literal.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(23) // integer constant is too large

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_zero(cuda::std::__fp_storage_t<Fmt> expected)
{
  assert(cuda::std::__fp_zero<Fmt>() == expected);
  static_assert(((void) cuda::std::__fp_zero<Fmt>(), true));
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_zero()
{}

template <class T,
          auto fmt                                                                  = cuda::std::__fp_format_of_v<T>,
          cuda::std::enable_if_t<fmt != cuda::std::__fp_format::__fp8_nv_e8m0, int> = 0>
__host__ __device__ void test_fp_zero()
{
  assert(cuda::std::__fp_get_storage(cuda::std::__fp_zero<T>()) == cuda::std::__fp_zero<fmt>());
  static_assert(((void) cuda::std::__fp_zero<T>(), true));
}

template <class T,
          auto fmt                                                                  = cuda::std::__fp_format_of_v<T>,
          cuda::std::enable_if_t<fmt == cuda::std::__fp_format::__fp8_nv_e8m0, int> = 0>
__host__ __device__ void test_fp_zero()
{}

__host__ __device__ bool test()
{
  using namespace test_integer_literals;

  // 1. Test formats
  test_fp_zero<cuda::std::__fp_format::__binary16>(0x0000u);
  test_fp_zero<cuda::std::__fp_format::__binary32>(0x00000000u);
  test_fp_zero<cuda::std::__fp_format::__binary64>(0x0000000000000000u);
#if _CCCL_HAS_INT128()
  test_fp_zero<cuda::std::__fp_format::__binary128>(0x00000000000000000000000000000000_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_zero<cuda::std::__fp_format::__bfloat16>(0x0000u);
#if _CCCL_HAS_INT128()
  test_fp_zero<cuda::std::__fp_format::__fp80_x86>(0x00000000000000000000_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_zero<cuda::std::__fp_format::__fp8_nv_e4m3>(0x00u);
  test_fp_zero<cuda::std::__fp_format::__fp8_nv_e5m2>(0x00u);
  test_fp_zero<cuda::std::__fp_format::__fp8_nv_e8m0>();
  test_fp_zero<cuda::std::__fp_format::__fp6_nv_e2m3>(0x00u);
  test_fp_zero<cuda::std::__fp_format::__fp6_nv_e3m2>(0x00u);
  test_fp_zero<cuda::std::__fp_format::__fp4_nv_e2m1>(0x0u);

  // 2. Test types
  test_fp_zero<float>();
  test_fp_zero<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_zero<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_zero<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_zero<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_zero<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_zero<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_zero<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_zero<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_zero<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_zero<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_fp_zero<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  // todo: test __cccl_fp types

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
