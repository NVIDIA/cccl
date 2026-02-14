//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license nansormation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(23) // integer constant is too large

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_nans(cuda::std::__fp_storage_t<Fmt> expected)
{
  assert(cuda::std::__fp_nans<Fmt>() == expected);
  static_assert(((void) cuda::std::__fp_nans<Fmt>(), true));
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_nans()
{
  static_assert(!cuda::std::__fp_has_nans_v<Fmt>);
}

template <class T, cuda::std::enable_if_t<cuda::std::__fp_has_nans_v<cuda::std::__fp_format_of_v<T>>, int> = 0>
__host__ __device__ void test_fp_nans()
{
  // constexpr auto fmt = cuda::std::__fp_format_of_v<T>;
  const auto result = cuda::std::__fp_nans<T>();
  assert(cuda::std::isnan(result));

  // todo: make this work, see issue #5555
  //   if constexpr (fmt == cuda::std::__fp_format::__fp8_nv_e4m3 || fmt == cuda::std::__fp_format::__fp8_nv_e5m2
  //                 || fmt == cuda::std::__fp_format::__fp8_nv_e8m0 || fmt == cuda::std::__fp_format::__fp6_nv_e2m3
  //                 || fmt == cuda::std::__fp_format::__fp6_nv_e3m2 || fmt == cuda::std::__fp_format::__fp4_nv_e2m1)
  //   {
  //     assert(cuda::std::__fp_get_storage(result) == cuda::std::__fp_nans<fmt>());
  //   }
  //   else
  //   {
  //     const auto nan_distinct_mask = cuda::std::__fp_storage_t<fmt>{1}
  //                                 << (cuda::std::__fp_mant_nbits_v<fmt> - 1 -
  //                                 !cuda::std::__fp_has_implicit_bit_v<fmt>);
  //     assert(!(cuda::std::__fp_get_storage(result) & nan_distinct_mask));
  //   }

  //   // NVRTC doesn't implement __builtin_nans, so we fallback to __builtin_bit_cast which is available since 12.3
  // #if !_CCCL_COMPILER(NVRTC, <, 12, 3)
  //   static_assert(((void) cuda::std::__fp_nans<T>(), true));
  // #endif // !_CCCL_COMPILER(NVRTC, <, 12, 3)
}

template <class T, cuda::std::enable_if_t<!cuda::std::__fp_has_nans_v<cuda::std::__fp_format_of_v<T>>, int> = 0>
__host__ __device__ void test_fp_nans()
{}

__host__ __device__ bool test()
{
  using namespace test_integer_literals;

  // 1. Test formats
  test_fp_nans<cuda::std::__fp_format::__binary16>(0x7d00u);
  test_fp_nans<cuda::std::__fp_format::__binary32>(0x7fa00000u);
  test_fp_nans<cuda::std::__fp_format::__binary64>(0x7ff4000000000000ull);
#if _CCCL_HAS_INT128()
  test_fp_nans<cuda::std::__fp_format::__binary128>(0x7fff4000000000000000000000000000_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_nans<cuda::std::__fp_format::__bfloat16>(0x7fa0u);
#if _CCCL_HAS_INT128()
  test_fp_nans<cuda::std::__fp_format::__fp80_x86>(0x7fffa000000000000000_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_nans<cuda::std::__fp_format::__fp8_nv_e4m3>();
  test_fp_nans<cuda::std::__fp_format::__fp8_nv_e5m2>(0x7du);
  test_fp_nans<cuda::std::__fp_format::__fp8_nv_e8m0>();
  test_fp_nans<cuda::std::__fp_format::__fp6_nv_e2m3>();
  test_fp_nans<cuda::std::__fp_format::__fp6_nv_e3m2>();
  test_fp_nans<cuda::std::__fp_format::__fp4_nv_e2m1>();

  // 2. Test types
  test_fp_nans<float>();
  test_fp_nans<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_nans<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_nans<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_nans<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_nans<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_nans<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_nans<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_nans<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_nans<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_nans<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_fp_nans<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  // todo: test __cccl_fp types

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
