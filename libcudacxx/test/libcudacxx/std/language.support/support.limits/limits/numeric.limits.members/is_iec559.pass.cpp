//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// is_iec559

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, bool expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::is_iec559 == expected, "is_iec559 test 1");
  static_assert(cuda::std::numeric_limits<const T>::is_iec559 == expected, "is_iec559 test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::is_iec559 == expected, "is_iec559 test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::is_iec559 == expected, "is_iec559 test 4");
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
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE
#if defined(_LIBCUDACXX_HAS_NVFP16)
  test<__half, true>();
#endif // _LIBCUDACXX_HAS_NVFP16
#if defined(_LIBCUDACXX_HAS_NVBF16)
  test<__nv_bfloat16, true>();
#endif // _LIBCUDACXX_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3, false>();
  test<__nv_fp8_e5m2, false>();
#endif // _CCCL_HAS_NVFP8()

  return 0;
}
