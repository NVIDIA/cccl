//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// test numeric_limits

// digits10

#include <cuda/std/cfloat>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, cuda::std::enable_if_t<cuda::std::is_integral<T>::value, int> = 0>
__host__ __device__ constexpr int make_expected_digits10()
{
  // digits * log10(2)
  return static_cast<int>((cuda::std::numeric_limits<T>::digits * 30103l) / 100000l);
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_integral<T>::value, int> = 0>
__host__ __device__ constexpr int make_expected_digits10()
{
  // (digits - 1) * log10(2)
  return static_cast<int>(((cuda::std::numeric_limits<T>::digits - 1) * 30103l) / 100000l);
}

template <class T, int expected = make_expected_digits10<T>()>
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
  test<bool>();
  test<char>();
  test<signed char>();
  test<unsigned char>();
  test<wchar_t>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<long>();
  test<unsigned long>();
  test<long long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
