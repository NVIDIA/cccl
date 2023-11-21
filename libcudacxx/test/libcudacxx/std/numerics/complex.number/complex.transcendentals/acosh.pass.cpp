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
//   acosh(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
    assert(acosh(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(INFINITY, 1), cuda::std::complex<T>(INFINITY, 0));
}

template <class T>
__host__ __device__ void test_edges()
{
    const T pi = cuda::std::atan2(+0., -0.);
    auto testcases = get_testcases<T>();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda::std::complex<T> r = acosh(testcases[i]);
        if (testcases[i].real() == T(0) && testcases[i].imag() == T(0))
        {
            assert(!cuda::std::signbit(r.real()));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/T(2));
            else
                is_about(r.imag(),  pi/T(2));
        }
        else if (testcases[i].real() == T(1) && testcases[i].imag() == T(0))
        {
            assert(r.real() == T(0));
            assert(!cuda::std::signbit(r.real()));
            assert(r.imag() == T(0));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == T(-1) && testcases[i].imag() == T(0))
        {
            assert(r.real() == T(0));
            assert(!cuda::std::signbit(r.real()));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi);
            else
                is_about(r.imag(),  pi);
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/T(2));
            else
                is_about(r.imag(),  pi/T(2));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi);
            else
                is_about(r.imag(),  pi);
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(r.imag() == T(0));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), T(-0.75) * pi);
            else
                is_about(r.imag(),  T(0.75) * pi);
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), T(-0.25) * pi);
            else
                is_about(r.imag(),  T(0.25) * pi);
        }
        else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else
        {
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test<__half>();
    test<__nv_bfloat16>();
    test_edges<double>();
    test_edges<__half>();
    test_edges<__nv_bfloat16>();

  return 0;
}
