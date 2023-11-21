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
//   T
//   norm(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

#include <iostream>

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    cuda::std::complex<T> z(3, 4);
    assert(norm(z) == T(25));

    return true;
}

template <class T>
__host__ __device__ void test_edges()
{
    auto testcases = get_testcases<T>();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        T r = norm(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
            assert(r == T(0));
            assert(!cuda::std::signbit(r));
            break;
        case non_zero:
            printf("%f %f\n", float(testcases[i].real()), float(testcases[i].imag()));
            assert(cuda::std::isfinite(r) && r > T(0));
            break;
        case inf:
            assert(cuda::std::isinf(r) && r > T(0));
            break;
        case NaN:
            assert(cuda::std::isnan(r));
            break;
        case non_zero_nan:
            assert(cuda::std::isnan(r));
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
#if TEST_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS)
    static_assert(test<float>(), "");
    static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif
    test<__half>();
    test<__nv_bfloat16>();

    test_edges<double>();
    test_edges<__half>();
    test_edges<__nv_bfloat16>();

  return 0;
}
