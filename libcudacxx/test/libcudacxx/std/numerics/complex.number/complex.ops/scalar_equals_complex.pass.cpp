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
//   bool
//   operator==(const T& lhs, const complex<T>& rhs);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
test_constexpr()
{
#if TEST_STD_VER > 11
    {
    constexpr T lhs(-2.5);
    constexpr cuda::std::complex<T> rhs(1.5,  2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr T lhs(-2.5);
    constexpr cuda::std::complex<T> rhs(1.5,  0);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr T lhs(1.5);
    constexpr cuda::std::complex<T> rhs(1.5, 2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr T lhs(1.5);
    constexpr cuda::std::complex<T> rhs(1.5, 0);
    static_assert(lhs == rhs, "");
    }
#endif
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    {
        T lhs(-2.5);
        cuda::std::complex<T> rhs(1.5,  2.5);
        assert(!(lhs == rhs));
    }
    {
        T lhs(-2.5);
        cuda::std::complex<T> rhs(1.5,  0);
        assert(!(lhs == rhs));
    }
    {
        T lhs(1.5);
        cuda::std::complex<T> rhs(1.5, 2.5);
        assert(!(lhs == rhs));
    }
    {
        T lhs(1.5);
        cuda::std::complex<T> rhs(1.5, 0);
        assert(lhs == rhs);
    }

    test_constexpr<T> ();

    return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
//     test_constexpr<int>();
#if TEST_STD_VER > 11
    static_assert(test<float>(), "");
    static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

  return 0;
}
