//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// digits10

#include <cuda/std/cfloat>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T, int expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::digits10 == expected, "digits10 test 1");
  static_assert(cuda::std::numeric_limits<T>::is_bounded, "digits10 test 5");
  static_assert(cuda::std::numeric_limits<const T>::digits10 == expected, "digits10 test 2");
  static_assert(cuda::std::numeric_limits<const T>::is_bounded, "digits10 test 6");
  static_assert(cuda::std::numeric_limits<volatile T>::digits10 == expected, "digits10 test 3");
  static_assert(cuda::std::numeric_limits<volatile T>::is_bounded, "digits10 test 7");
  static_assert(cuda::std::numeric_limits<const volatile T>::digits10 == expected, "digits10 test 4");
  static_assert(cuda::std::numeric_limits<const volatile T>::is_bounded, "digits10 test 8");
}

int main(int, char**)
{
  test<bool, 0>();
  test<char, 2>();
  test<signed char, 2>();
  test<unsigned char, 2>();
  test<wchar_t, 5 * sizeof(wchar_t) / 2 - 1>(); // 4 -> 9 and 2 -> 4
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, 2>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t, 4>();
  test<char32_t, 9>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short, 4>();
  test<unsigned short, 4>();
  test<int, 9>();
  test<unsigned int, 9>();
  test<long, sizeof(long) == 4 ? 9 : 18>();
  test<unsigned long, sizeof(long) == 4 ? 9 : 19>();
  test<long long, 18>();
  test<unsigned long long, 19>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t, 38>();
  test<__uint128_t, 38>();
#endif
  test<float, FLT_DIG>();
  test<double, DBL_DIG>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, LDBL_DIG>();
#endif

  return 0;
}
