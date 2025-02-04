//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/limits>

#include "test_macros.h"

/*
<limits>:
    numeric_limits
        is_specialized
        digits
        digits10
        max_digits10
        is_signed
        is_integer
        is_exact
        radix
        min_exponent
        min_exponent10
        max_exponent
        max_exponent10
        has_infinity
        has_quiet_NaN
        has_signaling_NaN
        has_denorm
        has_denorm_loss
        is_iec559
        is_bounded
        is_modulo
        traps
        tinyness_before
        round_style
*/

template <class T>
__host__ __device__ void test(T)
{}

template <class T>
__host__ __device__ void test_type_helper()
{
  test(cuda::std::numeric_limits<T>::is_specialized);
  test(cuda::std::numeric_limits<T>::digits);
  test(cuda::std::numeric_limits<T>::digits10);
  test(cuda::std::numeric_limits<T>::max_digits10);
  test(cuda::std::numeric_limits<T>::is_signed);
  test(cuda::std::numeric_limits<T>::is_integer);
  test(cuda::std::numeric_limits<T>::is_exact);
  test(cuda::std::numeric_limits<T>::radix);
  test(cuda::std::numeric_limits<T>::min_exponent);
  test(cuda::std::numeric_limits<T>::min_exponent10);
  test(cuda::std::numeric_limits<T>::max_exponent);
  test(cuda::std::numeric_limits<T>::max_exponent10);
  test(cuda::std::numeric_limits<T>::has_infinity);
  test(cuda::std::numeric_limits<T>::has_quiet_NaN);
  test(cuda::std::numeric_limits<T>::has_signaling_NaN);
  test(cuda::std::numeric_limits<T>::has_denorm);
  test(cuda::std::numeric_limits<T>::has_denorm_loss);
  test(cuda::std::numeric_limits<T>::is_iec559);
  test(cuda::std::numeric_limits<T>::is_bounded);
  test(cuda::std::numeric_limits<T>::is_modulo);
  test(cuda::std::numeric_limits<T>::traps);
  test(cuda::std::numeric_limits<T>::tinyness_before);
  test(cuda::std::numeric_limits<T>::round_style);
}

template <class T>
__host__ __device__ void test_type()
{
  test_type_helper<T>();
  test_type_helper<const T>();
  test_type_helper<volatile T>();
  test_type_helper<const volatile T>();
}

struct other
{};

int main(int, char**)
{
  test_type<bool>();
  test_type<char>();
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<wchar_t>();
#if TEST_STD_VER >= 2020 && defined(__cpp_char8_t)
  test_type<char8_t>();
#endif // TEST_STD_VER >= 2020 && defined(__cpp_char8_t)
  test_type<char16_t>();
  test_type<char32_t>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test_type<__int128_t>();
#endif // _LIBCUDACXX_HAS_NO_INT128
  test_type<float>();
  test_type<double>();
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test_type<long double>();
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE
#if defined(_CCCL_HAS_NVFP16)
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16
#if defined(_CCCL_HAS_NVBF16)
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test_type<__nv_fp8_e4m3>();
  test_type<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8()

  return 0;
}
