//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   bool
//   operator==(const complex<T>& lhs, const T& rhs);

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

#if defined(_MSC_VER)
#  pragma warning(disable : 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_constexpr()
{
  {
    constexpr cuda::std::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr cuda::std::complex<T> lhs(1.5, 0);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr cuda::std::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(1.5);
    static_assert(!(lhs == rhs), "");
  }
  {
    constexpr cuda::std::complex<T> lhs(1.5, 0);
    constexpr T rhs(1.5);
    static_assert((lhs == rhs), "");
  }
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_nonconstexpr()
{
  {
    cuda::std::complex<T> lhs(1.5, 2.5);
    T rhs(-2.5);
    assert(!(lhs == rhs));
  }
  {
    cuda::std::complex<T> lhs(1.5, 0);
    T rhs(-2.5);
    assert(!(lhs == rhs));
  }
  {
    cuda::std::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    assert(!(lhs == rhs));
  }
  {
    cuda::std::complex<T> lhs(1.5, 0);
    T rhs(1.5);
    assert((lhs == rhs));
  }
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_nonconstexpr<T>();
  test_constexpr<T>();

  return true;
}

int main(int, char**)
{
  test<float>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_nonconstexpr<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_nonconstexpr<__nv_bfloat16>();
#endif
  // CUDA treats long double as double
  //  test<long double>();
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
  // CUDA treats long double as double
  //  static_assert(test<long double>(), "");
  test_constexpr<int>();

  return 0;
}
