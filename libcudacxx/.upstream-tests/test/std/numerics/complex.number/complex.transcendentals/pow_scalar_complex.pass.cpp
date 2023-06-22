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
//   pow(const T& x, const complex<T>& y);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const T& a, const cuda::std::complex<T>& b, cuda::std::complex<T> x)
{
    cuda::std::complex<T> c = pow(a, b);
    is_about(real(c), real(x));
    assert(cuda::std::abs(imag(c)) < 1.e-6);
}

template <class T>
__host__ __device__ void
test()
{
    test(T(2), cuda::std::complex<T>(2), cuda::std::complex<T>(4));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            cuda::std::complex<double> r = pow(real(testcases[i]), testcases[j]);
            cuda::std::complex<double> z = exp(testcases[j] * log(cuda::std::complex<double>(real(testcases[i]))));
            if (cuda::std::isnan(real(r)))
                assert(cuda::std::isnan(real(z)));
            else
            {
                assert(real(r) == real(z));
            }
            if (cuda::std::isnan(imag(r)))
                assert(cuda::std::isnan(imag(z)));
            else
            {
                assert(imag(r) == imag(z));
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
    test_edges();

  return 0;
}
