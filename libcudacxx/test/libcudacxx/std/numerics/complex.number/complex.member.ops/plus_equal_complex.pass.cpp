//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// complex& operator+=(const complex& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  cuda::std::complex<T> c;
  const cuda::std::complex<T> c2(1.5, 2.5);
  assert(c.real() == T(0));
  assert(c.imag() == T(0));
  c += c2;
  assert(c.real() == T(1.5));
  assert(c.imag() == T(2.5));
  c += c2;
  assert(c.real() == T(3));
  assert(c.imag() == T(5));

  cuda::std::complex<T> c3;

  c3 = c;
  cuda::std::complex<int> ic(1, 1);
  c3 += ic;
  assert(c3.real() == T(4));
  assert(c3.imag() == T(6));

  c3 = c;
  cuda::std::complex<float> fc(1, 1);
  c3 += fc;
  assert(c3.real() == T(4));
  assert(c3.imag() == T(6));

  return true;
}

int main(int, char**)
{
  test<float>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif
// CUDA treats long double as double
//  test<long double>();
#if TEST_STD_VER > 2011
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

  return 0;
}
