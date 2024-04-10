//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// infinity()

#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/limits>

// MSVC has issues with producing INF with divisions by zero.
#if defined(_MSC_VER)
#  include <cmath>
#endif

#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(cuda::std::numeric_limits<T>::infinity() == expected);
  assert(cuda::std::numeric_limits<const T>::infinity() == expected);
  assert(cuda::std::numeric_limits<volatile T>::infinity() == expected);
  assert(cuda::std::numeric_limits<const volatile T>::infinity() == expected);
}

int main(int, char**)
{
  test<bool>(false);
  test<char>(0);
  test<signed char>(0);
  test<unsigned char>(0);
  test<wchar_t>(0);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(0);
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t>(0);
  test<char32_t>(0);
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short>(0);
  test<unsigned short>(0);
  test<int>(0);
  test<unsigned int>(0);
  test<long>(0);
  test<unsigned long>(0);
  test<long long>(0);
  test<unsigned long long>(0);
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t>(0);
  test<__uint128_t>(0);
#endif
#if !defined(_MSC_VER)
  test<float>(1.f / 0.f);
  test<double>(1. / 0.);
#  ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>(1. / 0.);
#  endif
// MSVC has issues with producing INF with divisions by zero.
#else
  test<float>(INFINITY);
  test<double>(INFINITY);
#  ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>(INFINITY);
#  endif
#endif

  return 0;
}

#ifndef TEST_COMPILER_NVRTC
float zero = 0;
#endif
