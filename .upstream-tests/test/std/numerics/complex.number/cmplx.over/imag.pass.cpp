//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<Arithmetic T>
//   T
//   imag(const T& x);

#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T, int x>
__host__ __device__ void
test(typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::imag(T(x))), double>::value), "");
    assert(cuda::std::imag(x) == 0);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(cuda::std::imag(val) == 0, "");
    constexpr cuda::std::complex<T> t{val, val};
    static_assert(t.imag() == x, "" );
#endif
}

template <class T, int x>
__host__ __device__ void
test(typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::imag(T(x))), T>::value), "");
    assert(cuda::std::imag(x) == 0);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(cuda::std::imag(val) == 0, "");
    constexpr cuda::std::complex<T> t{val, val};
    static_assert(t.imag() == x, "" );
#endif
}

template <class T>
__host__ __device__ void
test()
{
    test<T, 0>();
    test<T, 1>();
    test<T, 10>();
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
