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
#include <cuda/std/cassert>

template <class T>
__host__ __device__ void test_layout()
{
  cuda::complex<T> z{T(1), T(2)};
  T* p = (T*) &z;
  assert(T(1) == z.real());
  assert(T(2) == z.imag());
  assert(p[0] == z.real());
  assert(p[1] == z.imag());
  p[0] = T(2);
  p[1] = T(4);
  assert(p[0] == z.real());
  assert(p[1] == z.imag());
}

__host__ __device__ void test()
{
  test_layout<float>();
  test_layout<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_layout<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_layout<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_layout<signed char>();
  test_layout<signed short>();
  test_layout<signed int>();
  test_layout<signed long>();
  test_layout<signed long long>();
#if _CCCL_HAS_INT128()
  test_layout<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_layout<unsigned char>();
  test_layout<unsigned short>();
  test_layout<unsigned int>();
  test_layout<unsigned long>();
  test_layout<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_layout<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  return 0;
}
