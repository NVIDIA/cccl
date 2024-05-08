//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// make_signed

#include <cuda/std/type_traits>

#include "test_macros.h"

enum Enum
{
  zero,
  one_
};

enum BigEnum : unsigned long long // MSVC's ABI doesn't follow the Standard
{
  bigzero,
  big = 0xFFFFFFFFFFFFFFFFULL
};

#if !defined(_LIBCUDACXX_HAS_NO_INT128)
enum HugeEnum : __uint128_t
{
  hugezero
};
#endif // !_LIBCUDACXX_HAS_NO_INT128

template <class T, class U>
__host__ __device__ void test_make_signed()
{
  ASSERT_SAME_TYPE(U, typename cuda::std::make_signed<T>::type);
#if TEST_STD_VER > 2011
  ASSERT_SAME_TYPE(U, cuda::std::make_signed_t<T>);
#endif
}

int main(int, char**)
{
  test_make_signed<signed char, signed char>();
  test_make_signed<unsigned char, signed char>();
  test_make_signed<char, signed char>();
  test_make_signed<short, signed short>();
  test_make_signed<unsigned short, signed short>();
  test_make_signed<int, signed int>();
  test_make_signed<unsigned int, signed int>();
  test_make_signed<long, signed long>();
  test_make_signed<unsigned long, long>();
  test_make_signed<long long, signed long long>();
  test_make_signed<unsigned long long, signed long long>();
  test_make_signed<wchar_t, cuda::std::conditional<sizeof(wchar_t) == 4, int, short>::type>();
  test_make_signed<const wchar_t, cuda::std::conditional<sizeof(wchar_t) == 4, const int, const short>::type>();
  test_make_signed<const Enum, cuda::std::conditional<sizeof(Enum) == sizeof(int), const int, const signed char>::type>();
  test_make_signed<BigEnum, cuda::std::conditional<sizeof(long) == 4, long long, long>::type>();
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test_make_signed<__int128_t, __int128_t>();
  test_make_signed<__uint128_t, __int128_t>();
  test_make_signed<HugeEnum, __int128_t>();
#endif // !_LIBCUDACXX_HAS_NO_INT128

  return 0;
}
