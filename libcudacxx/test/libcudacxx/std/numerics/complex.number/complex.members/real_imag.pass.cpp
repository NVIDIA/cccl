//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// void real(T val);
// void imag(T val);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_constexpr()
{
#if TEST_STD_VER > 2011
  constexpr cuda::std::complex<T> c1;
  static_assert(c1.real() == 0, "");
  static_assert(c1.imag() == 0, "");
  constexpr cuda::std::complex<T> c2(3);
  static_assert(c2.real() == 3, "");
  static_assert(c2.imag() == 0, "");
  constexpr cuda::std::complex<T> c3(3, 4);
  static_assert(c3.real() == 3, "");
  static_assert(c3.imag() == 4, "");
#endif
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_nonconstexpr()
{
  cuda::std::complex<T> c;
  assert(c.real() == T(0));
  assert(c.imag() == T(0));
  c.real(3.5);
  assert(c.real() == T(3.5));
  assert(c.imag() == T(0));
  c.imag(4.5);
  assert(c.real() == T(3.5));
  assert(c.imag() == T(4.5));
  c.real(-4.5);
  assert(c.real() == T(-4.5));
  assert(c.imag() == T(4.5));
  c.imag(-5.5);
  assert(c.real() == T(-4.5));
  assert(c.imag() == T(-5.5));
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_nonconstexpr<T>();
  test_constexpr<T>();

  return true;
}

template <class T>
__host__ __device__ void test_volatile()
{
  volatile cuda::std::complex<T> cv;
  assert(cv.real() == T(0));
  assert(cv.imag() == T(0));
  cv.real(3.5);
  assert(cv.real() == T(3.5));
  assert(cv.imag() == T(0));
  cv.imag(4.5);
  assert(cv.real() == T(3.5));
  assert(cv.imag() == T(4.5));
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_nonconstexpr<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_nonconstexpr<__nv_bfloat16>();
#endif

#if TEST_STD_VER > 2011
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif
  test_constexpr<int>();

  // test volatile extensions
  test_volatile<float>();
  test_volatile<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  // test_volatile<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  // test_volatile<__nv_bfloat16>();
#endif

  return 0;
}
