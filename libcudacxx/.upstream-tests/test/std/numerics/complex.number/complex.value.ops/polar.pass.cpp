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
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const T& rho, cuda::std::complex<T> x)
{
    assert(cuda::std::polar(rho) == x);
}

template <class T>
__host__ __device__ void
test(const T& rho, const T& theta, cuda::std::complex<T> x)
{
    assert(cuda::std::polar(rho, theta) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(T(0), cuda::std::complex<T>(0, 0));
    test(T(1), cuda::std::complex<T>(1, 0));
    test(T(100), cuda::std::complex<T>(100, 0));
    test(T(0), T(0), cuda::std::complex<T>(0, 0));
    test(T(1), T(0), cuda::std::complex<T>(1, 0));
    test(T(100), T(0), cuda::std::complex<T>(100, 0));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = real(testcases[i]);
        double theta = imag(testcases[i]);
        cuda::std::complex<double> z = cuda::std::polar(r, theta);
        switch (classify(r))
        {
        case zero:
            if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                assert(z == cuda::std::complex<double>());
            }
            break;
        case non_zero:
            if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                is_about(cuda::std::abs(z), r);
            }
            break;
        case inf:
            if (r < 0)
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
            else
            {
                assert(classify(z) == inf);
                if (classify(theta) != NaN && classify(theta) != inf)
                {
                    assert(classify(real(z)) != NaN);
                    assert(classify(imag(z)) != NaN);
                }
            }
            break;
        case NaN:
        case non_zero_nan:
            {
                int c = classify(z);
                assert(c == NaN || c == non_zero_nan);
            }
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
    test_edges();

  return 0;
}
