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
//   conj(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(const cuda::std::complex<T>& z, cuda::std::complex<T> x)
{
  assert(conj(z) == x);
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test(cuda::std::complex<T>(1, 2), cuda::std::complex<T>(1, -2));
  test(cuda::std::complex<T>(-1, 2), cuda::std::complex<T>(-1, -2));
  test(cuda::std::complex<T>(1, -2), cuda::std::complex<T>(1, 2));
  test(cuda::std::complex<T>(-1, -2), cuda::std::complex<T>(-1, 2));

  return true;
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

#if TEST_STD_VER > 2011
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

  return 0;
}
