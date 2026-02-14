//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license maxormation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(23) // integer constant is too large

template <cuda::std::__fp_format Fmt>
__host__ __device__ void test_fp_max(cuda::std::__fp_storage_t<Fmt> expected)
{
  assert(cuda::std::__fp_max<Fmt>() == expected);
  static_assert(((void) cuda::std::__fp_max<Fmt>(), true));
}

template <class T>
__host__ __device__ void test_fp_max()
{
  constexpr auto fmt = cuda::std::__fp_format_of_v<T>;
  assert(cuda::std::__fp_get_storage(cuda::std::__fp_max<T>()) == cuda::std::__fp_max<fmt>());
  static_assert(((void) cuda::std::__fp_max<T>(), true));
}

__host__ __device__ bool test()
{
  using namespace test_integer_literals;

  // 1. Test formats
  test_fp_max<cuda::std::__fp_format::__binary16>(0x7bffu);
  test_fp_max<cuda::std::__fp_format::__binary32>(0x7f7fffffu);
  test_fp_max<cuda::std::__fp_format::__binary64>(0x7fefffffffffffffull);
#if _CCCL_HAS_INT128()
  test_fp_max<cuda::std::__fp_format::__binary128>(0x7ffeffffffffffffffffffffffffffff_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_max<cuda::std::__fp_format::__bfloat16>(0x7f7fu);
#if _CCCL_HAS_INT128()
  test_fp_max<cuda::std::__fp_format::__fp80_x86>(0x7ffeffffffffffffffff_u128);
#endif // _CCCL_HAS_INT128()
  test_fp_max<cuda::std::__fp_format::__fp8_nv_e4m3>(0x7eu);
  test_fp_max<cuda::std::__fp_format::__fp8_nv_e5m2>(0x7bu);
  test_fp_max<cuda::std::__fp_format::__fp8_nv_e8m0>(0xfeu);
  test_fp_max<cuda::std::__fp_format::__fp6_nv_e2m3>(0x1fu);
  test_fp_max<cuda::std::__fp_format::__fp6_nv_e3m2>(0x1fu);
  test_fp_max<cuda::std::__fp_format::__fp4_nv_e2m1>(0x7u);

  // 2. Test types
  test_fp_max<float>();
  test_fp_max<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_max<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_max<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_max<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_max<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_max<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_max<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_max<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_max<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_max<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_fp_max<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  // todo: test __cccl_fp types

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
