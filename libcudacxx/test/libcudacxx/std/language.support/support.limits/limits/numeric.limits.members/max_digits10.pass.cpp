//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// max_digits10

#include <cuda/std/cfloat>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, cuda::std::enable_if_t<cuda::std::is_integral<T>::value, int> = 0>
__host__ __device__ constexpr int make_expected_max_digits10()
{
  return 0;
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_integral<T>::value, int> = 0>
__host__ __device__ constexpr int make_expected_max_digits10()
{
  // std::ceil(std::numeric_limits<float>::digits * std::log10(2) + 1)
  return static_cast<int>((cuda::std::numeric_limits<T>::digits * 30103l + 99999l) / 100000l) + 1;
}

template <class T, int expected = make_expected_max_digits10<T>()>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::max_digits10 == expected, "max_digits10 test 1");
  static_assert(cuda::std::numeric_limits<const T>::max_digits10 == expected, "max_digits10 test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::max_digits10 == expected, "max_digits10 test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::max_digits10 == expected, "max_digits10 test 4");
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
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3>();
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8()

  return 0;
}
