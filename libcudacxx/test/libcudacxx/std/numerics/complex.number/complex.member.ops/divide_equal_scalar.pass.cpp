//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// complex& operator/=(const T& rhs);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    cuda::std::complex<T> c(1);
    assert(c.real() == T(1));
    assert(c.imag() == T(0));
    c /= 0.5;
    assert(c.real() == T(2));
    assert(c.imag() == T(0));
    c /= 0.5;
    assert(c.real() == T(4));
    assert(c.imag() == T(0));
    c /= -0.5;
    assert(c.real() == T(-8));
    assert(c.imag() == T(0));
    c.imag(2);
    c /= 0.5;
    assert(c.real() == T(-16));
    assert(c.imag() == T(4));

    return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
#ifndef _LIBCUDACXX_HAS_NO_NVFP16
    test<__half>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
    test<__nv_bfloat16>();
#endif
// CUDA treats long double as double
//  test<long double>();
#if TEST_STD_VER > 2011
    static_assert(test<float>(), "");
    static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

  return 0;
}
