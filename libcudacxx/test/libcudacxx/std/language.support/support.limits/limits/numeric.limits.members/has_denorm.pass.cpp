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
#if _CCCL_HAS_INT128()
  test<__int128_t, cuda::std::denorm_absent>();
  test<__uint128_t, cuda::std::denorm_absent>();
#endif // _CCCL_HAS_INT128()
  test<float, cuda::std::denorm_present>();
  test<double, cuda::std::denorm_present>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double, cuda::std::denorm_present>();
#endif
#if _CCCL_HAS_NVFP16()
  test<__half, cuda::std::denorm_present>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, cuda::std::denorm_present>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3, cuda::std::denorm_present>();
  test<__nv_fp8_e5m2, cuda::std::denorm_present>();
#  if _CCCL_CUDACC_AT_LEAST(12, 8)
  test<__nv_fp8_e8m0, cuda::std::denorm_absent>();
#  endif // _CCCL_CUDACC_AT_LEAST(12, 8)
#endif // _CCCL_HAS_NVFP8()
#if _CCCL_HAS_NVFP6()
  test<__nv_fp6_e2m3, cuda::std::denorm_present>();
  test<__nv_fp6_e3m2, cuda::std::denorm_present>();
#endif // _CCCL_HAS_NVFP6
#if _CCCL_HAS_NVFP4()
  test<__nv_fp4_e2m1, cuda::std::denorm_present>();
#endif // _CCCL_HAS_NVFP4

  return 0;
}
