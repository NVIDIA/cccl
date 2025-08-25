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
#include <cuda/std/type_traits>

template <class T>
__host__ __device__ void test_types()
{
  static_assert(cuda::std::is_same_v<T, typename cuda::complex<T>::value_type>);
}

__host__ __device__ void test()
{
  test_types<float>();
  test_types<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_types<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_types<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_types<signed char>();
  test_types<signed short>();
  test_types<signed int>();
  test_types<signed long>();
  test_types<signed long long>();
#if _CCCL_HAS_INT128()
  test_types<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_types<unsigned char>();
  test_types<unsigned short>();
  test_types<unsigned int>();
  test_types<unsigned long>();
  test_types<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_types<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
