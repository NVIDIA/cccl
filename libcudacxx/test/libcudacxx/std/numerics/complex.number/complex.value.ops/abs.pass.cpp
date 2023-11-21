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
//   abs(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test()
{
    cuda::std::complex<T> z(3, 4);
    assert(abs(z) == T(5));
}

template <class T>
__host__ __device__ void test_edges()
{
    auto testcases = get_testcases<T>();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        T r = abs(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
            assert(r == T(0));
            assert(!cuda::std::signbit(r));
            break;
        case non_zero:
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
    test<__half>();
    test<__nv_bfloat16>();
    test_edges<double>();
    test_edges<__half>();
    test_edges<__nv_bfloat16>();

  return 0;
}
