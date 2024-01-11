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
//   asinh(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
    assert(asinh(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(0, 0));
}

__host__ __device__ void test_edges()
{
    const double pi = cuda::std::atan2(+0., -0.);
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda::std::complex<double> r = asinh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(cuda::std::signbit(r.real()) == cuda::std::signbit(testcases[i].real()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == 0 && cuda::std::abs(testcases[i].imag()) == 1)
        {
            assert(r.real() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/2);
            else
                is_about(r.imag(),  pi/2);
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::signbit(testcases[i].real()) == cuda::std::signbit(r.real()));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/2);
            else
                is_about(r.imag(),  pi/2);
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::signbit(testcases[i].real()) == cuda::std::signbit(r.real()));
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::signbit(testcases[i].real()) == cuda::std::signbit(r.real()));
            if (cuda::std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/4);
            else
                is_about(r.imag(),  pi/4);
        }
        else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::signbit(testcases[i].real()) == cuda::std::signbit(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
            assert(cuda::std::isnan(r.real()));
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else
        {
            assert(cuda::std::signbit(r.real()) == cuda::std::signbit(testcases[i].real()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }

        const auto asinh_conj = cuda::std::asinh(cuda::std::conj(testcases[i]));
        const auto conj_asinh = cuda::std::conj(cuda::std::asinh(testcases[i]));
        if (asinh_conj != conj_asinh) {
            if (cuda::std::isnan(asinh_conj.real())) {
                assert(cuda::std::isnan(conj_asinh.real()));
            } else if (cuda::std::isinf(asinh_conj.real())) {
                assert(cuda::std::isinf(conj_asinh.real()));
            } else if (cuda::std::isnan(asinh_conj.imag())) {
                assert(cuda::std::isnan(conj_asinh.imag()));
            } else if (cuda::std::isinf(asinh_conj.imag())) {
                assert(cuda::std::isinf(conj_asinh.imag()));
            } else {
                assert(false);
            }
        }

        const auto neg_asinh = -cuda::std::asinh(testcases[i]);
        const auto asinh_neg = cuda::std::asinh(-testcases[i]);
        if (neg_asinh != asinh_neg) {
            if (cuda::std::isnan(neg_asinh.real())) {
                assert(cuda::std::isnan(asinh_neg.real()));
            } else if (cuda::std::isinf(neg_asinh.real())) {
                assert(cuda::std::isinf(asinh_neg.real()));
            } else if (cuda::std::isnan(neg_asinh.imag())) {
                assert(cuda::std::isnan(asinh_neg.imag()));
            } else if (cuda::std::isinf(neg_asinh.imag())) {
                assert(cuda::std::isinf(asinh_neg.imag()));
            } else {
                assert(cuda::std::abs((neg_asinh.real()-asinh_neg.real())) < 1.e-3);
                assert(cuda::std::abs((neg_asinh.imag()-asinh_neg.imag())) < 1.e-3);
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
