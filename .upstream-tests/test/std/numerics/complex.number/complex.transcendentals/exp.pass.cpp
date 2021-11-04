//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   exp(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
    assert(exp(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(1, 0));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda::std::complex<double> r = exp(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(r.real() == 1.0);
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && testcases[i].imag() == 0)
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda::std::isinf(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(r.imag() == 0);
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda::std::isnan(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(r.imag() == 0);
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
            assert(cuda::std::isnan(r.real()));
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && testcases[i].imag() != 0)
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isfinite(testcases[i].imag()) && cuda::std::abs(testcases[i].imag()) <= 1)
        {
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (cuda::std::isinf(r.real()) && testcases[i].imag() == 0) {
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
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
