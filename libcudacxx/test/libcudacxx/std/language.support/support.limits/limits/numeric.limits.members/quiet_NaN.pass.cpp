//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// quiet_NaN()

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_imp(cuda::std::true_type)
{
  assert(cuda::std::isnan(cuda::std::numeric_limits<T>::quiet_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<const T>::quiet_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<volatile T>::quiet_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<const volatile T>::quiet_NaN()));
}

template <class T>
__host__ __device__ void test_imp(cuda::std::false_type)
{
  assert(cuda::std::numeric_limits<T>::quiet_NaN() == T());
  assert(cuda::std::numeric_limits<const T>::quiet_NaN() == T());
  assert(cuda::std::numeric_limits<volatile T>::quiet_NaN() == T());
  assert(cuda::std::numeric_limits<const volatile T>::quiet_NaN() == T());
}

template <class T>
__host__ __device__ inline void test()
{
  constexpr bool is_float = cuda::std::is_floating_point<T>::value || cuda::std::__is_extended_floating_point<T>::value;

  test_imp<T>(cuda::std::integral_constant<bool, is_float>{});
}

int main(int, char**)
{
  test<bool>();
  test<char>();
  test<signed char>();
  test<unsigned char>();
  test<wchar_t>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t>();
  test<char32_t>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<long>();
  test<unsigned long>();
  test<long long>();
  test<unsigned long long>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t>();
  test<__uint128_t>();
#endif
  test<float>();
  test<double>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>();
#endif
#if defined(_LIBCUDACXX_HAS_NVFP16)
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16
#if defined(_LIBCUDACXX_HAS_NVBF16)
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16

  return 0;
}
