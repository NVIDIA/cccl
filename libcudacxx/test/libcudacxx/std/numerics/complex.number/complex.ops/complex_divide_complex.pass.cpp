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
//   complex<T>
//   operator/(const complex<T>& lhs, const complex<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  cuda::std::complex<T> lhs(-4.0, 7.5);
  cuda::std::complex<T> rhs(1.5, 2.5);
  cuda::std::complex<T> x(1.5, 2.5);
  assert(lhs / rhs == x);

  return true;
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    for (unsigned j = 0; j < N; ++j)
    {
      cuda::std::complex<T> r = testcases[i] / testcases[j];
      switch (classify(testcases[i]))
      {
        case zero:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == NaN);
              break;
            case non_zero:
              assert(classify(r) == zero);
              break;
            case inf:
              assert(classify(r) == zero);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case non_zero:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == inf);
              break;
            case non_zero:
              assert(classify(r) == non_zero);
              break;
            case inf:
              assert(classify(r) == zero);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case inf:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == inf);
              break;
            case non_zero:
              assert(classify(r) == inf);
              break;
            case inf:
              assert(classify(r) == NaN);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case NaN:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == NaN);
              break;
            case non_zero:
              assert(classify(r) == NaN);
              break;
            case inf:
              assert(classify(r) == NaN);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case non_zero_nan:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == inf);
              break;
            case non_zero:
              assert(classify(r) == NaN);
              break;
            case inf:
              assert(classify(r) == NaN);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
      }
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#if TEST_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS)
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif

  test_edges<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_edges<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_edges<__nv_bfloat16>();
#endif

  return 0;
}
