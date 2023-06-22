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

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    cuda::std::complex<T> z(3, 4);
    assert(norm(z) == 25);

    return true;
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = norm(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
            assert(r == 0);
            assert(!cuda::std::signbit(r));
            break;
        case non_zero:
            assert(cuda::std::isfinite(r) && r > 0);
            break;
        case inf:
            assert(cuda::std::isinf(r) && r > 0);
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
#if TEST_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS)
    static_assert(test<float>(), "");
    static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

    test_edges();

  return 0;
}
