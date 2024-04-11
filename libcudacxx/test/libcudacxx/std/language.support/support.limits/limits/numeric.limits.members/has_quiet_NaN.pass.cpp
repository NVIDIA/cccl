//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// has_quiet_NaN

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, bool expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::has_quiet_NaN == expected, "has_quiet_NaN test 1");
  static_assert(cuda::std::numeric_limits<const T>::has_quiet_NaN == expected, "has_quiet_NaN test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::has_quiet_NaN == expected, "has_quiet_NaN test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::has_quiet_NaN == expected, "has_quiet_NaN test 4");
}

int main(int, char**)
{
  test<bool, false>();
  test<char, false>();
  test<signed char, false>();
  test<unsigned char, false>();
  test<wchar_t, false>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, false>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t, false>();
  test<char32_t, false>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short, false>();
  test<unsigned short, false>();
  test<int, false>();
  test<unsigned int, false>();
  test<long, false>();
  test<unsigned long, false>();
  test<long long, false>();
  test<unsigned long long, false>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t, false>();
  test<__uint128_t, false>();
#endif
  test<float, true>();
  test<double, true>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, true>();
#endif

  return 0;
}
