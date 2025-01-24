//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// quiet_NaN()

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ bool is_nan(T x)
{
  return cuda::std::isnan(x);
}

#if _CCCL_HAS_NVFP8()
__host__ __device__ bool is_nan(__nv_fp8_e4m3 x)
{
  return is_nan(__half{__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3)});
}

__host__ __device__ bool is_nan(__nv_fp8_e5m2 x)
{
  return is_nan(__half{__nv_cvt_fp8_to_halfraw(x.__x, __NV_E5M2)});
}
#endif // _CCCL_HAS_NVFP8()

template <class T>
__host__ __device__ void test_impl(cuda::std::true_type)
{
  assert(is_nan(cuda::std::numeric_limits<T>::quiet_NaN()));
  assert(is_nan(cuda::std::numeric_limits<const T>::quiet_NaN()));
  assert(is_nan(cuda::std::numeric_limits<volatile T>::quiet_NaN()));
  assert(is_nan(cuda::std::numeric_limits<const volatile T>::quiet_NaN()));
}

template <class T>
__host__ __device__ bool equal_to(T x, T y)
{
  return x == y;
}

#if _CCCL_HAS_NVFP8()
__host__ __device__ bool equal_to(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y)
{
  return x.__x == y.__x;
}

__host__ __device__ bool equal_to(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP8()

template <class T>
__host__ __device__ void test_impl(cuda::std::false_type)
{
  assert(equal_to(cuda::std::numeric_limits<T>::signaling_NaN(), T()));
  assert(equal_to(cuda::std::numeric_limits<const T>::signaling_NaN(), T()));
  assert(equal_to(cuda::std::numeric_limits<volatile T>::signaling_NaN(), T()));
  assert(equal_to(cuda::std::numeric_limits<const volatile T>::signaling_NaN(), T()));
}

template <class T>
__host__ __device__ inline void test()
{
  test_impl<T>(cuda::std::integral_constant<bool, cuda::std::numeric_limits<T>::has_quiet_NaN>{});
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
