//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// max()

#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/climits>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(cuda::std::numeric_limits<T>::max() == expected);
  assert(cuda::std::numeric_limits<T>::is_bounded);
  assert(cuda::std::numeric_limits<const T>::max() == expected);
  assert(cuda::std::numeric_limits<const T>::is_bounded);
  assert(cuda::std::numeric_limits<volatile T>::max() == expected);
  assert(cuda::std::numeric_limits<volatile T>::is_bounded);
  assert(cuda::std::numeric_limits<const volatile T>::max() == expected);
  assert(cuda::std::numeric_limits<const volatile T>::is_bounded);
}

int main(int, char**)
{
#ifndef TEST_COMPILER_NVRTC
  test<wchar_t>(WCHAR_MAX);
#endif
  test<bool>(true);
  test<char>(CHAR_MAX);
  test<signed char>(SCHAR_MAX);
  test<unsigned char>(UCHAR_MAX);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(UCHAR_MAX); // ??
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t>(USHRT_MAX);
  test<char32_t>(UINT_MAX);
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short>(SHRT_MAX);
  test<unsigned short>(USHRT_MAX);
  test<int>(INT_MAX);
  test<unsigned int>(UINT_MAX);
  test<long>(LONG_MAX);
  test<unsigned long>(ULONG_MAX);
  test<long long>(LLONG_MAX);
  test<unsigned long long>(ULLONG_MAX);
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t>(__int128_t(__uint128_t(-1) / 2));
  test<__uint128_t>(__uint128_t(-1));
#endif
  test<float>(FLT_MAX);
  test<double>(DBL_MAX);
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>(LDBL_MAX);
#endif

  return 0;
}
