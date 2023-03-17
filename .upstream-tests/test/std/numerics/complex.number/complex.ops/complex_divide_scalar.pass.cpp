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
//   operator/(const complex<T>& lhs, const T& rhs);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& lhs, const T& rhs, cuda::std::complex<T> x)
{
    assert(lhs / rhs == x);
}

template <class T>
__host__ __device__ void
test()
{
    cuda::std::complex<T> lhs(-4.0, 7.5);
    T rhs(2);
    cuda::std::complex<T>   x(-2, 3.75);
    test(lhs, rhs, x);
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();

  return 0;
}
