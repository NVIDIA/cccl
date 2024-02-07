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
//   sqrt(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
    cuda::std::complex<T> a = sqrt(c);
    is_about(real(a), real(x));
    assert(cuda::std::abs(imag(c)) < T(1.e-6));
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(64, 0), cuda::std::complex<T>(8, 0));
}

template <class T>
__host__ __device__ void test_edges()
{
    auto testcases = get_testcases<T>();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda::std::complex<T> r = sqrt(testcases[i]);
        if (testcases[i].real() == T(0) && testcases[i].imag() == T(0))
        {
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == T(0));
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(r.imag() == T(0));
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < T(0) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isinf(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > T(0) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isinf(r.real()));
            assert(r.real() > T(0));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && (cuda::std::isfinite(testcases[i].imag()) || cuda::std::isnan(testcases[i].imag())))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::signbit(testcases[i].imag()))
        {
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::signbit(r.imag()));
        }
        else
        {
            assert(!cuda::std::signbit(r.real()));
            assert(!cuda::std::signbit(r.imag()));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifndef _LIBCUDACXX_HAS_NO_NVFP16
    test<__half>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
    test<__nv_bfloat16>();
#endif
    test_edges<double>();
#ifndef _LIBCUDACXX_HAS_NO_NVFP16
    test_edges<__half>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
    test_edges<__nv_bfloat16>();
#endif

  return 0;
}
