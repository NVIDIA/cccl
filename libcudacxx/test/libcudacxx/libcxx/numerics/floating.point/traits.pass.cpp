//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__floating_point/fp.h>

template <class T>
__host__ __device__ void test_is_std_fp()
{
  static_assert(cuda::std::__is_std_fp_v<T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<T>);
  static_assert(!cuda::std::__is_ext_fp_v<T>);
  static_assert(cuda::std::__is_fp_v<T>);

  static_assert(cuda::std::__is_std_fp_v<const T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<const T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<const T>);
  static_assert(!cuda::std::__is_ext_fp_v<const T>);
  static_assert(cuda::std::__is_fp_v<const T>);

  static_assert(cuda::std::__is_std_fp_v<volatile T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<volatile T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<volatile T>);
  static_assert(!cuda::std::__is_ext_fp_v<volatile T>);
  static_assert(cuda::std::__is_fp_v<volatile T>);

  static_assert(cuda::std::__is_std_fp_v<const volatile T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<const volatile T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<const volatile T>);
  static_assert(!cuda::std::__is_ext_fp_v<const volatile T>);
  static_assert(cuda::std::__is_fp_v<const volatile T>);
}

template <class T>
__host__ __device__ void test_ext_nv_fp()
{
  static_assert(!cuda::std::__is_std_fp_v<T>);
  static_assert(cuda::std::__is_ext_nv_fp_v<T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<T>);
  static_assert(cuda::std::__is_ext_fp_v<T>);
  static_assert(cuda::std::__is_fp_v<T>);

  static_assert(!cuda::std::__is_std_fp_v<const T>);
  static_assert(cuda::std::__is_ext_nv_fp_v<const T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<const T>);
  static_assert(cuda::std::__is_ext_fp_v<const T>);
  static_assert(cuda::std::__is_fp_v<const T>);

  static_assert(!cuda::std::__is_std_fp_v<volatile T>);
  static_assert(cuda::std::__is_ext_nv_fp_v<volatile T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<volatile T>);
  static_assert(cuda::std::__is_ext_fp_v<volatile T>);
  static_assert(cuda::std::__is_fp_v<volatile T>);

  static_assert(!cuda::std::__is_std_fp_v<const volatile T>);
  static_assert(cuda::std::__is_ext_nv_fp_v<const volatile T>);
  static_assert(!cuda::std::__is_ext_compiler_fp_v<const volatile T>);
  static_assert(cuda::std::__is_ext_fp_v<const volatile T>);
  static_assert(cuda::std::__is_fp_v<const volatile T>);
}

template <class T>
__host__ __device__ void test_ext_compiler_fp()
{
  static_assert(!cuda::std::__is_std_fp_v<T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<T>);
  static_assert(cuda::std::__is_ext_compiler_fp_v<T>);
  static_assert(cuda::std::__is_ext_fp_v<T>);
  static_assert(cuda::std::__is_fp_v<T>);

  static_assert(!cuda::std::__is_std_fp_v<const T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<const T>);
  static_assert(cuda::std::__is_ext_compiler_fp_v<const T>);
  static_assert(cuda::std::__is_ext_fp_v<const T>);
  static_assert(cuda::std::__is_fp_v<const T>);

  static_assert(!cuda::std::__is_std_fp_v<volatile T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<volatile T>);
  static_assert(cuda::std::__is_ext_compiler_fp_v<volatile T>);
  static_assert(cuda::std::__is_ext_fp_v<volatile T>);
  static_assert(cuda::std::__is_fp_v<volatile T>);

  static_assert(!cuda::std::__is_std_fp_v<const volatile T>);
  static_assert(!cuda::std::__is_ext_nv_fp_v<const volatile T>);
  static_assert(cuda::std::__is_ext_compiler_fp_v<const volatile T>);
  static_assert(cuda::std::__is_ext_fp_v<const volatile T>);
  static_assert(cuda::std::__is_fp_v<const volatile T>);
}

int main(int, char**)
{
  test_is_std_fp<float>();
  test_is_std_fp<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_is_std_fp<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
  test_ext_nv_fp<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_ext_nv_fp<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_ext_nv_fp<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_ext_nv_fp<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_ext_nv_fp<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_ext_nv_fp<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_ext_nv_fp<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_ext_nv_fp<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

#if _CCCL_HAS_FLOAT128()
  test_ext_compiler_fp<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
