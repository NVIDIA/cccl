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
//   cos(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& c,
                              cuda::std::complex<T> x) {
  assert(cos(c) == x);
}

template <class T>
__host__ __device__ void test() {
  test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(1, 0));
}

template <class T>
__host__ __device__ void test_edges() {
  auto testcases = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i) {
    cuda::std::complex<T> r = cos(testcases[i]);
    cuda::std::complex<T> t1(-imag(testcases[i]), real(testcases[i]));
    cuda::std::complex<T> z = cosh(t1);
    if (cuda::std::isnan(real(r)))
      assert(cuda::std::isnan(real(z)));
    else {
      assert(real(r) == real(z));
      assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(z)));
    }
    if (cuda::std::isnan(imag(r)))
      assert(cuda::std::isnan(imag(z)));
    else {
      assert(imag(r) == imag(z));
      assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(z)));
    }
  }
}

int main(int, char**) {
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
