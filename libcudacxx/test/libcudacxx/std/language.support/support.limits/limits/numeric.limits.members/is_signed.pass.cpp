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

// is_signed

#include <cuda/std/limits>

#include "test_macros.h"

template <class T, bool expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::is_signed == expected, "is_signed test 1");
  static_assert(cuda::std::numeric_limits<const T>::is_signed == expected, "is_signed test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::is_signed == expected, "is_signed test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::is_signed == expected, "is_signed test 4");
}

int main(int, char**)
{
  test<bool, false>();
  test<char, char(-1) < char(0)>();
  test<signed char, true>();
  test<unsigned char, false>();
  test<wchar_t, wchar_t(-1) < wchar_t(0)>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, false>();
#endif
  test<char16_t, false>();
  test<char32_t, false>();
  test<short, true>();
  test<unsigned short, false>();
  test<int, true>();
  test<unsigned int, false>();
  test<long, true>();
  test<unsigned long, false>();
  test<long long, true>();
  test<unsigned long long, false>();
#if _CCCL_HAS_INT128()
  test<__int128_t, true>();
  test<__uint128_t, false>();
#endif // _CCCL_HAS_INT128()
  test<float, true>();
  test<double, true>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double, true>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half, true>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, true>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3, true>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2, true>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0, false>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3, true>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2, true>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1, true>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128, true>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
