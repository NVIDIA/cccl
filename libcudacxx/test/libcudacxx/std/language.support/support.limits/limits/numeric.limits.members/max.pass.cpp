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

// max()

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(float_eq(cuda::std::numeric_limits<T>::max(), expected));
  assert(cuda::std::numeric_limits<T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<const T>::max(), expected));
  assert(cuda::std::numeric_limits<const T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<volatile T>::max(), expected));
  assert(cuda::std::numeric_limits<volatile T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::max(), expected));
  assert(cuda::std::numeric_limits<const volatile T>::is_bounded);
}

int main(int, char**)
{
#if !TEST_COMPILER(NVRTC)
  test<wchar_t>(WCHAR_MAX);
#endif // !TEST_COMPILER(NVRTC)
  test<bool>(true);
  test<char>(CHAR_MAX);
  test<signed char>(SCHAR_MAX);
  test<unsigned char>(UCHAR_MAX);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(UCHAR_MAX); // ??
#endif
  test<char16_t>(USHRT_MAX);
  test<char32_t>(UINT_MAX);
  test<short>(SHRT_MAX);
  test<unsigned short>(USHRT_MAX);
  test<int>(INT_MAX);
  test<unsigned int>(UINT_MAX);
  test<long>(LONG_MAX);
  test<unsigned long>(ULONG_MAX);
  test<long long>(LLONG_MAX);
  test<unsigned long long>(ULLONG_MAX);
#if _CCCL_HAS_INT128()
  test<__int128_t>(__int128_t(__uint128_t(-1) / 2));
  test<__uint128_t>(__uint128_t(-1));
#endif // _CCCL_HAS_INT128()
  test<float>(FLT_MAX);
  test<double>(DBL_MAX);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(LDBL_MAX);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>(__double2half(65504.0));
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__double2bfloat16(3.3895313892515355e+38));
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>(make_fp8_e4m3(448.0));
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>(make_fp8_e5m2(57344.0));
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>(make_fp8_e8m0(3.40282366920938463463374607431768211456e+38));
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>(make_fp6_e2m3(7.5));
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>(make_fp6_e3m2(28.0));
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>(make_fp4_e2m1(6.0));
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128>(cuda::std::bit_cast<__float128>((__uint128_t{0x7ffe'ffff'ffff'ffff} << 64) | 0xffff'ffff'ffff'ffff));
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
