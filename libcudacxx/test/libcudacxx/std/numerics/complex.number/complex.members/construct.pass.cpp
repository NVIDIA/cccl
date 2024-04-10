//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// constexpr complex(const T& re = T(), const T& im = T());

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_constexpr()
{
  {
    constexpr cuda::std::complex<T> c;
    static_assert(c.real() == 0, "");
    static_assert(c.imag() == 0, "");
  }
  {
    constexpr cuda::std::complex<T> c = 7.5;
    static_assert(c.real() == 7.5, "");
    static_assert(c.imag() == 0, "");
  }
  {
    constexpr cuda::std::complex<T> c(8.5);
    static_assert(c.real() == 8.5, "");
    static_assert(c.imag() == 0, "");
  }
  {
    constexpr cuda::std::complex<T> c(10.5, -9.5);
    static_assert(c.real() == 10.5, "");
    static_assert(c.imag() == -9.5, "");
  }
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_nonconstexpr()
{
  {
    const cuda::std::complex<T> c;
    assert(c.real() == T(0));
    assert(c.imag() == T(0));
  }
  {
    const cuda::std::complex<T> c = T(7.5);
    assert(c.real() == T(7.5));
    assert(c.imag() == T(0));
  }
  {
    const cuda::std::complex<T> c(8.5);
    assert(c.real() == T(8.5));
    assert(c.imag() == T(0));
  }
  {
    const cuda::std::complex<T> c(10.5, -9.5);
    assert(c.real() == T(10.5));
    assert(c.imag() == T(-9.5));
  }
}

template <class T>
__host__ __device__ void test()
{
  test_nonconstexpr<T>();
  test_constexpr<T>();
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

  return 0;
}
