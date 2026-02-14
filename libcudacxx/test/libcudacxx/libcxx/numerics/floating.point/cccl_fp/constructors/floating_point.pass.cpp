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
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

template <cuda::std::__fp_format Fmt, class Fp>
__host__ __device__ constexpr void test_fp_constructor()
{
  using T = cuda::std::__cccl_fp<Fmt>;

  // Construction from a floating point type is always noexcept
  static_assert(cuda::std::is_nothrow_constructible_v<T, Fp>);

  // Construction from a floating point type is implicit if T has the greater or equal conversion rank
  static_assert(cuda::std::__fp_is_implicit_conversion_v<Fp, T> == cuda::std::is_convertible_v<Fp, T>);
  static_assert(cuda::std::__fp_is_explicit_conversion_v<Fp, T> == !cuda::std::is_convertible_v<Fp, T>);

  // TODO: check construction from a floating point type
  [[maybe_unused]] T val{Fp{}};
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ constexpr void test_format()
{
  // standard floating point types
  test_fp_constructor<Fmt, float>();
  test_fp_constructor<Fmt, double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_constructor<Fmt, long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  // extended nvidia floating point types
#if _CCCL_HAS_NVFP16()
  test_fp_constructor<Fmt, __half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_constructor<Fmt, __nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_constructor<Fmt, __nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_constructor<Fmt, __nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_constructor<Fmt, __nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_constructor<Fmt, __nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_constructor<Fmt, __nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_constructor<Fmt, __nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  // extended compiler floating point types
#if _CCCL_HAS_FLOAT128()
  test_fp_constructor<Fmt, __float128>();
#endif // _CCCL_HAS_FLOAT128()

  // extended cccl floating point types
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__binary16>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__binary32>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__binary64>>();
#if _CCCL_HAS_INT128()
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__binary128>>();
#endif // _CCCL_HAS_INT128()
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__bfloat16>>();
#if _CCCL_HAS_INT128()
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp80_x86>>();
#endif // _CCCL_HAS_INT128()
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e4m3>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e5m2>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp8_nv_e8m0>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e2m3>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp6_nv_e3m2>>();
  test_fp_constructor<Fmt, cuda::std::__cccl_fp<cuda::std::__fp_format::__fp4_nv_e2m1>>();
}

__host__ __device__ constexpr bool test()
{
  test_format<cuda::std::__fp_format::__binary16>();
  test_format<cuda::std::__fp_format::__binary32>();
  test_format<cuda::std::__fp_format::__binary64>();
#if _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__binary128>();
#endif // _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__bfloat16>();
#if _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__fp80_x86>();
#endif // _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__fp8_nv_e4m3>();
  test_format<cuda::std::__fp_format::__fp8_nv_e5m2>();
  test_format<cuda::std::__fp_format::__fp8_nv_e8m0>();
  test_format<cuda::std::__fp_format::__fp6_nv_e2m3>();
  test_format<cuda::std::__fp_format::__fp6_nv_e3m2>();
  test_format<cuda::std::__fp_format::__fp4_nv_e2m1>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
