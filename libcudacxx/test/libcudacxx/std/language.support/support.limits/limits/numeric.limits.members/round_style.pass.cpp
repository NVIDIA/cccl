//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// round_style

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, cuda::std::float_round_style expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::round_style == expected, "round_style test 1");
  static_assert(cuda::std::numeric_limits<const T>::round_style == expected, "round_style test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::round_style == expected, "round_style test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::round_style == expected, "round_style test 4");
}

int main(int, char**)
{
  test<bool, cuda::std::round_toward_zero>();
  test<char, cuda::std::round_toward_zero>();
  test<signed char, cuda::std::round_toward_zero>();
  test<unsigned char, cuda::std::round_toward_zero>();
  test<wchar_t, cuda::std::round_toward_zero>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, cuda::std::round_toward_zero>();
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t, cuda::std::round_toward_zero>();
  test<char32_t, cuda::std::round_toward_zero>();
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short, cuda::std::round_toward_zero>();
  test<unsigned short, cuda::std::round_toward_zero>();
  test<int, cuda::std::round_toward_zero>();
  test<unsigned int, cuda::std::round_toward_zero>();
  test<long, cuda::std::round_toward_zero>();
  test<unsigned long, cuda::std::round_toward_zero>();
  test<long long, cuda::std::round_toward_zero>();
  test<unsigned long long, cuda::std::round_toward_zero>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t, cuda::std::round_toward_zero>();
  test<__uint128_t, cuda::std::round_toward_zero>();
#endif
  test<float, cuda::std::round_to_nearest>();
  test<double, cuda::std::round_to_nearest>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, cuda::std::round_to_nearest>();
#endif
#if defined(_LIBCUDACXX_HAS_NVFP16)
  test<__half, cuda::std::round_to_nearest>();
#endif // _LIBCUDACXX_HAS_NVFP16
#if defined(_LIBCUDACXX_HAS_NVBF16)
  test<__nv_bfloat16, cuda::std::round_to_nearest>();
#endif // _LIBCUDACXX_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3, cuda::std::round_to_nearest>();
  test<__nv_fp8_e5m2, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP8()

  return 0;
}
