//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   proj(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& z, cuda::std::complex<T> x)
{
  assert(proj(z) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(1, 2), cuda::std::complex<T>(1, 2));
  test(cuda::std::complex<T>(-1, 2), cuda::std::complex<T>(-1, 2));
  test(cuda::std::complex<T>(1, -2), cuda::std::complex<T>(1, -2));
  test(cuda::std::complex<T>(-1, -2), cuda::std::complex<T>(-1, -2));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = proj(testcases[i]);
    switch (classify(testcases[i]))
    {
      case zero:
      case non_zero:
        assert(r == testcases[i]);
        assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(testcases[i])));
        assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(testcases[i])));
        break;
      case inf:
        assert(cuda::std::isinf(real(r)) && real(r) > T(0));
        assert(imag(r) == T(0));
        assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(testcases[i])));
        break;
      case NaN:
      case non_zero_nan:
        assert(classify(r) == classify(testcases[i]));
        break;
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
