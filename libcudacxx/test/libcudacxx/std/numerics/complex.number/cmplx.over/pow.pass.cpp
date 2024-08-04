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
#  pragma warning(disable : 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../cases.h"
#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test(T x, const cuda::std::complex<U>& y)
{
  using promote_t = typename cuda::std::common_type<T, U>::type;
  static_assert((cuda::std::is_same<decltype(cuda::std::pow(x, y)), cuda::std::complex<promote_t>>::value), "");
  assert(cuda::std::pow(x, y) == pow(cuda::std::complex<promote_t>(x, 0), cuda::std::complex<promote_t>(y)));
}

template <class T, class U>
__host__ __device__ void test(const cuda::std::complex<T>& x, U y)
{
  using promote_t = typename cuda::std::common_type<T, U>::type;
  static_assert((cuda::std::is_same<decltype(cuda::std::pow(x, y)), cuda::std::complex<promote_t>>::value), "");
  assert(cuda::std::pow(x, y) == pow(cuda::std::complex<promote_t>(x), cuda::std::complex<promote_t>(y, 0)));
}

template <class T, class U>
__host__ __device__ void test(const cuda::std::complex<T>& x, const cuda::std::complex<U>& y)
{
  using promote_t = typename cuda::std::common_type<T, U>::type;
  assert(cuda::std::pow(x, y) == pow(cuda::std::complex<promote_t>(x), cuda::std::complex<promote_t>(y)));
}

template <class T, class U>
__host__ __device__ void test(typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type*  = 0,
                              typename cuda::std::enable_if<!cuda::std::is_integral<U>::value>::type* = 0)
{
  test(T(3), cuda::std::complex<U>(4, 5));
  test(cuda::std::complex<U>(3, 4), T(5));
}

template <class T, class U>
__host__ __device__ void test(typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0,
                              typename cuda::std::enable_if<!cuda::std::is_integral<U>::value>::type* = 0)
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

#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half, float>();
  test<__half, double>();
  test<int, __half>();
  test<unsigned, __half>();
  test<long long, __half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16, float>();
  test<__nv_bfloat16, double>();
  test<int, __nv_bfloat16>();
  test<unsigned, __nv_bfloat16>();
  test<long long, __nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16

  return 0;
}
