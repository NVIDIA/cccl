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

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& a, const cuda::std::complex<T>& b, cuda::std::complex<T> x)
{
    cuda::std::complex<T> c = pow(a, b);
    is_about(real(c), real(x));
    is_about(imag(c), imag(x));
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(2, 3), cuda::std::complex<T>(2, 0), cuda::std::complex<T>(-5, 12));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            cuda::std::complex<double> r = pow(testcases[i], testcases[j]);
            cuda::std::complex<double> z = exp(testcases[j] * log(testcases[i]));
            if (cuda::std::isnan(real(r)))
                assert(cuda::std::isnan(real(z)));
            else
            {
                assert(real(r) == real(z));
                assert(cuda::std::signbit(real(r)) == cuda::std::signbit(real(z)));
            }
            if (cuda::std::isnan(imag(r)))
                assert(cuda::std::isnan(imag(z)));
            else
            {
                assert(imag(r) == imag(z));
                assert(cuda::std::signbit(imag(r)) == cuda::std::signbit(imag(z)));
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
