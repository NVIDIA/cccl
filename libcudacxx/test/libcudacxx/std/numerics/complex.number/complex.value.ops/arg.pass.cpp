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
//   T
//   arg(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  cuda::std::complex<T> z(1, 0);
  assert(arg(z) == T(0));
}

template <class T>
__host__ __device__ void test_edges()
{
  const T pi       = cuda::std::atan2(+0., -0.);
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    T r = arg(testcases[i]);
    if (cuda::std::isnan(testcases[i].real()) || cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r));
    }
    else
    {
      switch (classify(testcases[i]))
      {
        case zero:
          if (cuda::std::signbit(testcases[i].real()))
          {
            if (cuda::std::signbit(testcases[i].imag()))
            {
              is_about(r, -pi);
            }
            else
            {
              is_about(r, pi);
            }
          }
          else
          {
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r));
          }
          break;
        case non_zero:
          if (testcases[i].real() == T(0))
          {
            if (testcases[i].imag() < T(0))
            {
              is_about(r, -pi / T(2));
            }
            else
            {
              is_about(r, pi / T(2));
            }
          }
          else if (testcases[i].imag() == T(0))
          {
            if (testcases[i].real() < T(0))
            {
              if (cuda::std::signbit(testcases[i].imag()))
              {
                is_about(r, -pi);
              }
              else
              {
                is_about(r, pi);
              }
            }
            else
            {
              assert(r == T(0));
              assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r));
            }
          }
          else if (testcases[i].imag() > T(0))
          {
            assert(r > T(0));
          }
          else
          {
            assert(r < T(0));
          }
          break;
        case inf:
          if (cuda::std::isinf(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
          {
            if (testcases[i].real() < T(0))
            {
              if (testcases[i].imag() > T(0))
              {
                is_about(r, T(0.75) * pi);
              }
              else
              {
                is_about(r, T(-0.75) * pi);
              }
            }
            else
            {
              if (testcases[i].imag() > T(0))
              {
                is_about(r, T(0.25) * pi);
              }
              else
              {
                is_about(r, T(-0.25) * pi);
              }
            }
          }
          else if (cuda::std::isinf(testcases[i].real()))
          {
            if (testcases[i].real() < T(0))
            {
              if (cuda::std::signbit(testcases[i].imag()))
              {
                is_about(r, -pi);
              }
              else
              {
                is_about(r, pi);
              }
            }
            else
            {
              assert(r == T(0));
              assert(cuda::std::signbit(r) == cuda::std::signbit(testcases[i].imag()));
            }
          }
          else
          {
            if (testcases[i].imag() < T(0))
            {
              is_about(r, -pi / T(2));
            }
            else
            {
              is_about(r, pi / T(2));
            }
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
