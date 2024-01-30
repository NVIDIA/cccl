//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _LIBCUDACXX_HAS_NO_NVFP16
__host__ __device__ __half promote(__half);
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
__host__ __device__ __nv_bfloat16 promote(__nv_bfloat16);
#endif
#endif
__host__ __device__ float promote(float);
__host__ __device__ double promote(double);
__host__ __device__ long double promote(long double);

#ifndef _LIBCUDACXX_HAS_NO_NVFP16
// This is a workaround for __half's conversions being just bad.
__host__ __device__ float operator+(float, __half);
__host__ __device__ float operator+(__half, float);
__host__ __device__ double operator+(double, __half);
__host__ __device__ double operator+(__half, double);
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
// This is a workaround for __nv_bfloat16's conversions being just bad.
__host__ __device__ float operator+(float, __nv_bfloat16);
__host__ __device__ float operator+(__nv_bfloat16, float);
__host__ __device__ double operator+(double, __nv_bfloat16);
__host__ __device__ double operator+(__nv_bfloat16, double);
#endif
#endif

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

#ifndef _LIBCUDACXX_HAS_NO_NVFP16
    test<__half, float>();
    test<__half, double>();
    test<int, __half>();
    test<unsigned, __half>();
    test<long long, __half>();
#ifndef _LIBCUDACXX_HAS_NO_NVBF16
    test<__nv_bfloat16, float>();
    test<__nv_bfloat16, double>();
    test<int, __nv_bfloat16>();
    test<unsigned, __nv_bfloat16>();
    test<long long, __nv_bfloat16>();
#endif
#endif

  return 0;
}
