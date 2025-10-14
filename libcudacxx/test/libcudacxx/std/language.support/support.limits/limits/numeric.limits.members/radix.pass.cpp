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

// radix

#include <cuda/std/cfloat>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T, int expected>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::radix == expected, "radix test 1");
  static_assert(cuda::std::numeric_limits<const T>::radix == expected, "radix test 2");
  static_assert(cuda::std::numeric_limits<volatile T>::radix == expected, "radix test 3");
  static_assert(cuda::std::numeric_limits<const volatile T>::radix == expected, "radix test 4");
}

int main(int, char**)
{
  test<bool, 2>();
  test<char, 2>();
  test<signed char, 2>();
  test<unsigned char, 2>();
  test<wchar_t, 2>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t, 2>();
#endif
  test<char16_t, 2>();
  test<char32_t, 2>();
  test<short, 2>();
  test<unsigned short, 2>();
  test<int, 2>();
  test<unsigned int, 2>();
  test<long, 2>();
  test<unsigned long, 2>();
  test<long long, 2>();
  test<unsigned long long, 2>();
#if _CCCL_HAS_INT128()
  test<__int128_t, 2>();
  test<__uint128_t, 2>();
#endif // _CCCL_HAS_INT128()
  test<float, FLT_RADIX>();
  test<double, FLT_RADIX>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double, FLT_RADIX>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, FLT_RADIX>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1, FLT_RADIX>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128, FLT_RADIX>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
