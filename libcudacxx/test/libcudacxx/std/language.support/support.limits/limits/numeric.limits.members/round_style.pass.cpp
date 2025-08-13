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
  test<char16_t, cuda::std::round_toward_zero>();
  test<char32_t, cuda::std::round_toward_zero>();
  test<short, cuda::std::round_toward_zero>();
  test<unsigned short, cuda::std::round_toward_zero>();
  test<int, cuda::std::round_toward_zero>();
  test<unsigned int, cuda::std::round_toward_zero>();
  test<long, cuda::std::round_toward_zero>();
  test<unsigned long, cuda::std::round_toward_zero>();
  test<long long, cuda::std::round_toward_zero>();
  test<unsigned long long, cuda::std::round_toward_zero>();
#if _CCCL_HAS_INT128()
  test<__int128_t, cuda::std::round_toward_zero>();
  test<__uint128_t, cuda::std::round_toward_zero>();
#endif // _CCCL_HAS_INT128()
  test<float, cuda::std::round_to_nearest>();
  test<double, cuda::std::round_to_nearest>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0, cuda::std::round_toward_zero>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128, cuda::std::round_to_nearest>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
