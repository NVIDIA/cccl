//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// complex& operator=(const complex&);
// template<class X> complex& operator= (const complex<X>&);

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T, class X>
__host__ __device__ void
test()
{
    cuda::std::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    cuda::std::complex<T> c2(1.5, 2.5);
    c = c2;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
    cuda::std::complex<X> c3(3.5, -4.5);
    c = c3;
    assert(c.real() == 3.5);
    assert(c.imag() == -4.5);
}

int main(int, char**)
{
    test<float, float>();
    test<float, double>();

    test<double, float>();
    test<double, double>();

// CUDA treats long double as double
//  test<float, long double>();
//  test<double, long double>();
//  test<long double, float>();
//  test<long double, double>();
//  test<long double, long double>();


  return 0;
}
