//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const T& x, const complex<U>& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const U& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const complex<U>& y);

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ double
promote(T, typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0);

__host__ __device__ float promote(float);
__host__ __device__ double promote(double);
__host__ __device__ long double promote(long double);

template <class T, class U>
__host__ __device__ void
test(T x, const cuda::std::complex<U>& y)
{
    typedef decltype(promote(x)+promote(real(y))) V;
    static_assert((cuda::std::is_same<decltype(cuda::std::pow(x, y)), cuda::std::complex<V> >::value), "");
    assert(cuda::std::pow(x, y) == pow(cuda::std::complex<V>(x, 0), cuda::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(const cuda::std::complex<T>& x, U y)
{
    typedef decltype(promote(real(x))+promote(y)) V;
    static_assert((cuda::std::is_same<decltype(cuda::std::pow(x, y)), cuda::std::complex<V> >::value), "");
    assert(cuda::std::pow(x, y) == pow(cuda::std::complex<V>(x), cuda::std::complex<V>(y, 0)));
}

template <class T, class U>
__host__ __device__ void
test(const cuda::std::complex<T>& x, const cuda::std::complex<U>& y)
{
    typedef decltype(promote(real(x))+promote(real(y))) V;
    static_assert((cuda::std::is_same<decltype(cuda::std::pow(x, y)), cuda::std::complex<V> >::value), "");
    assert(cuda::std::pow(x, y) == pow(cuda::std::complex<V>(x), cuda::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0, typename cuda::std::enable_if<!cuda::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), cuda::std::complex<U>(4, 5));
    test(cuda::std::complex<U>(3, 4), T(5));
}

template <class T, class U>
__host__ __device__ void
test(typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0, typename cuda::std::enable_if<!cuda::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), cuda::std::complex<U>(4, 5));
    test(cuda::std::complex<T>(3, 4), U(5));
    test(cuda::std::complex<T>(3, 4), cuda::std::complex<U>(5, 6));
}

int main(int, char**)
{
    test<int, float>();
    test<int, double>();

    test<unsigned, float>();
    test<unsigned, double>();

    test<long long, float>();
    test<long long, double>();

    test<float, double>();

    test<double, float>();

// CUDA treats long double as double
//  test<int, long double>();
//  test<unsigned, long double>();
//  test<long long, long double>();
//  test<float, long double>();
//  test<double, long double>();
//  test<long double, float>();
//  test<long double, double>();

  return 0;
}
