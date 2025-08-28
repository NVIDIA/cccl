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

// <cuda/complex>

#include <cuda/__complex_>

template <class T>
__host__ __device__ void test_abi()
{
  static_assert(sizeof(cuda::complex<T>) == (sizeof(T) * 2), "wrong size");
  static_assert(alignof(cuda::complex<T>) == (alignof(T) * 2), "misaligned");
}

__host__ __device__ void test()
{
  test_abi<float>();
  test_abi<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_abi<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_abi<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_abi<signed char>();
  test_abi<signed short>();
  test_abi<signed int>();
  test_abi<signed long>();
  test_abi<signed long long>();
#if _CCCL_HAS_INT128()
  test_abi<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_abi<unsigned char>();
  test_abi<unsigned short>();
  test_abi<unsigned int>();
  test_abi<unsigned long>();
  test_abi<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_abi<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
