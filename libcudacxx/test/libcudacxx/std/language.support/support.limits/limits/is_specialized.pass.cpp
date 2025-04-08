//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// Specializations shall be provided for each arithmetic type, both floating
// point and integer, including bool. The member is_specialized shall be
// true for all such specializations of numeric_limits.

// Non-arithmetic standard types, such as complex<T> (26.3.2), shall not
// have specializations.

// From [numeric.limits]:

// The value of each member of a specialization of numeric_limits on a cv
// -qualified type cv T shall be equal to the value of the corresponding
// member of the specialization on the unqualified type T.

// More convenient to test it here.

#include <cuda/std/complex>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  static_assert(cuda::std::numeric_limits<T>::is_specialized, "cuda::std::numeric_limits<T>::is_specialized");
  static_assert(cuda::std::numeric_limits<const T>::is_specialized,
                "cuda::std::numeric_limits<const T>::is_specialized");
  static_assert(cuda::std::numeric_limits<volatile T>::is_specialized,
                "cuda::std::numeric_limits<volatile T>::is_specialized");
  static_assert(cuda::std::numeric_limits<const volatile T>::is_specialized,
                "cuda::std::numeric_limits<const volatile T>::is_specialized");
}

int main(int, char**)
{
  test<bool>();
  test<char>();
  test<wchar_t>();
  test<char16_t>();
  test<char32_t>();
  test<signed char>();
  test<unsigned char>();
  test<signed short>();
  test<unsigned short>();
  test<signed int>();
  test<unsigned int>();
  test<signed long>();
  test<unsigned long>();
  test<signed long long>();
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

  static_assert(!cuda::std::numeric_limits<cuda::std::complex<double>>::is_specialized,
                "!cuda::std::numeric_limits<cuda::std::complex<double> >::is_specialized");

  return 0;
}
