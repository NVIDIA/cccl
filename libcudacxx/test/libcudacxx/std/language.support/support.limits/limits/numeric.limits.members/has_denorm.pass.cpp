//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// has_denorm

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, cuda::std::float_denorm_style expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::has_denorm == expected, "has_denorm test 1");
  static_assert(cuda::std::numeric_limits<const T>::has_denorm == expected, "has_denorm test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::has_denorm == expected, "has_denorm test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::has_denorm == expected, "has_denorm test 4");
}

int main(int, char**)
{
  test<bool, cuda::std::denorm_absent>();
  test<char, cuda::std::denorm_absent>();
  test<signed char, cuda::std::denorm_absent>();
  test<unsigned char, cuda::std::denorm_absent>();
  test<wchar_t, cuda::std::denorm_absent>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, cuda::std::denorm_absent>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t, cuda::std::denorm_absent>();
  test<char32_t, cuda::std::denorm_absent>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short, cuda::std::denorm_absent>();
  test<unsigned short, cuda::std::denorm_absent>();
  test<int, cuda::std::denorm_absent>();
  test<unsigned int, cuda::std::denorm_absent>();
  test<long, cuda::std::denorm_absent>();
  test<unsigned long, cuda::std::denorm_absent>();
  test<long long, cuda::std::denorm_absent>();
  test<unsigned long long, cuda::std::denorm_absent>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t, cuda::std::denorm_absent>();
  test<__uint128_t, cuda::std::denorm_absent>();
#endif
  test<float, cuda::std::denorm_present>();
  test<double, cuda::std::denorm_present>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, cuda::std::denorm_present>();
#endif

  return 0;
}
