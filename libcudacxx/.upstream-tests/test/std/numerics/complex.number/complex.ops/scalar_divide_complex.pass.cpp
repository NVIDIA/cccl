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
//   operator/(const T& lhs, const complex<T>& rhs);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    T lhs(-8.5);
    cuda::std::complex<T> rhs(1.5, 2.5);
    cuda::std::complex<T>   x(-1.5, 2.5);
    assert(lhs / rhs == x);

    return true;
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

  return 0;
}
