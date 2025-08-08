//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/std/numbers>

#include <cuda/std/numbers>
#include <cuda/std/type_traits>

#include <test_macros.h>

template <class ExpectedT, class T>
__host__ __device__ constexpr bool test_defined(const T& value)
{
  static_assert(cuda::std::is_same_v<ExpectedT, T>);

  const ExpectedT* addr = &value;
  unused(addr);

  return true;
}

template <class T>
__host__ __device__ constexpr bool test_type()
{
  test_defined<T>(cuda::std::numbers::e_v<T>);
  test_defined<T>(cuda::std::numbers::log2e_v<T>);
  test_defined<T>(cuda::std::numbers::log10e_v<T>);
  test_defined<T>(cuda::std::numbers::pi_v<T>);
  test_defined<T>(cuda::std::numbers::inv_pi_v<T>);
  test_defined<T>(cuda::std::numbers::inv_sqrtpi_v<T>);
  test_defined<T>(cuda::std::numbers::ln2_v<T>);
  test_defined<T>(cuda::std::numbers::ln10_v<T>);
  test_defined<T>(cuda::std::numbers::sqrt2_v<T>);
  test_defined<T>(cuda::std::numbers::sqrt3_v<T>);
  test_defined<T>(cuda::std::numbers::inv_sqrt3_v<T>);
  test_defined<T>(cuda::std::numbers::egamma_v<T>);
  test_defined<T>(cuda::std::numbers::phi_v<T>);

  return true;
}

__host__ __device__ constexpr bool test()
{
  test_defined<double>(cuda::std::numbers::e);
  test_defined<double>(cuda::std::numbers::log2e);
  test_defined<double>(cuda::std::numbers::log10e);
  test_defined<double>(cuda::std::numbers::pi);
  test_defined<double>(cuda::std::numbers::inv_pi);
  test_defined<double>(cuda::std::numbers::inv_sqrtpi);
  test_defined<double>(cuda::std::numbers::ln2);
  test_defined<double>(cuda::std::numbers::ln10);
  test_defined<double>(cuda::std::numbers::sqrt2);
  test_defined<double>(cuda::std::numbers::sqrt3);
  test_defined<double>(cuda::std::numbers::inv_sqrt3);
  test_defined<double>(cuda::std::numbers::egamma);
  test_defined<double>(cuda::std::numbers::phi);

  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test_type<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  return true;
}

__global__ void test_kernel()
{
  test();
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
