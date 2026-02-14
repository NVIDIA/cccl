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
//   pow(const complex<T>& x, const complex<T>& y);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T, class U = T>
__host__ __device__ void test(const cuda::std::complex<T>& a, const cuda::std::complex<U>& b, cuda::std::complex<T> x)
{
  static_assert(cuda::std::is_same<decltype(pow(a, b)), cuda::std::complex<T>>::value, "");
  cuda::std::complex<T> c = pow(a, b);
  is_about(real(c), real(x));
  is_about(imag(c), imag(x));
}

template <class T, class U = T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(2, 3), cuda::std::complex<U>(2, 0), cuda::std::complex<T>(-5, 12));
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
      cuda::std::complex<T> r = cuda::std::pow(testcases[i], testcases[j]);
      cuda::std::complex<T> z = cuda::std::exp(testcases[j] * cuda::std::log(testcases[i]));

      // The __half or __nv_float16 functions use fp32, we need to account for this
      // as we are checking for floating-point equality:
#if _LIBCUDACXX_HAS_NVFP16()
      if constexpr (cuda::std::is_same_v<T, __half>)
      {
        z = cuda::std::exp<float>(
          cuda::std::complex<float>(testcases[j]) * cuda::std::log<float>(cuda::std::complex<float>(testcases[i])));
      }
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
      if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>)
      {
        z = cuda::std::exp<float>(
          cuda::std::complex<float>(testcases[j]) * cuda::std::log<float>(cuda::std::complex<float>(testcases[i])));
      }
#endif // _LIBCUDACXX_HAS_NVBF16()

      if (cuda::std::isnan(real(r)))
      {
        assert(cuda::std::isnan(real(z)));
      }
      else
      {
        if (real(r) != real(z))
        {
          is_about(real(r), real(z));
        }
        assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(z)));
      }
      if (cuda::std::isnan(imag(r)))
      {
        assert(cuda::std::isnan(imag(z)));
      }
      else
      {
        if (imag(r) != imag(z))
        {
          is_about(imag(r), imag(z));
        }
        assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(z)));
      }
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  // Also test conversions
  test<float, int>();
  test<double, size_t>();

  test_edges<double>();

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
