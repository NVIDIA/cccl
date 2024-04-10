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
//   acos(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
  assert(acos(c) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(INFINITY, 1), cuda::std::complex<T>(0, -INFINITY));
}

template <class T>
__host__ __device__ void test_edges()
{
  const T pi       = cuda::std::atan2(+0., -0.);
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = acos(testcases[i]);
    if (testcases[i].real() == T(0) && testcases[i].imag() == T(0))
    {
      is_about(r.real(), pi / T(2));
      assert(r.imag() == T(0));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (testcases[i].real() == T(0) && cuda::std::isnan(testcases[i].imag()))
    {
      is_about(r.real(), pi / T(2));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
    {
      is_about(r.real(), pi / T(2));
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && testcases[i].real() != T(0)
             && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0)
             && cuda::std::isfinite(testcases[i].imag()))
    {
      is_about(r.real(), pi);
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0)
             && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(r.real() == T(0));
      assert(!cuda::std::signbit(r.real()));
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0)
             && cuda::std::isinf(testcases[i].imag()))
    {
      is_about(r.real(), T(0.75) * pi);
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0)
             && cuda::std::isinf(testcases[i].imag()))
    {
      is_about(r.real(), T(0.25) * pi);
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isinf(r.imag()));
    }
    else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isinf(r.imag()));
      assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (!cuda::std::signbit(testcases[i].real()) && !cuda::std::signbit(testcases[i].imag()))
    {
      assert(!cuda::std::signbit(r.real()));
      assert(cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::signbit(testcases[i].real()) && !cuda::std::signbit(testcases[i].imag()))
    {
      assert(!cuda::std::signbit(r.real()));
      assert(cuda::std::signbit(r.imag()));
    }
    else if (cuda::std::signbit(testcases[i].real()) && cuda::std::signbit(testcases[i].imag()))
    {
      assert(!cuda::std::signbit(r.real()));
      assert(!cuda::std::signbit(r.imag()));
    }
    else if (!cuda::std::signbit(testcases[i].real()) && cuda::std::signbit(testcases[i].imag()))
    {
      assert(!cuda::std::signbit(r.real()));
      assert(!cuda::std::signbit(r.imag()));
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
