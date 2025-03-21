//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// make_unsigned

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

#if _CCCL_HAS_INT128()
enum HugeEnum : __int128_t
{
  hugezero
};
#endif // _CCCL_HAS_INT128()

template <class T, class U>
__host__ __device__ void test_make_unsigned()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::make_unsigned<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::make_unsigned_t<T>>);
}

int main(int, char**)
{
  test_make_unsigned<signed char, unsigned char>();
  test_make_unsigned<unsigned char, unsigned char>();
  test_make_unsigned<char, unsigned char>();
  test_make_unsigned<short, unsigned short>();
  test_make_unsigned<unsigned short, unsigned short>();
  test_make_unsigned<int, unsigned int>();
  test_make_unsigned<unsigned int, unsigned int>();
  test_make_unsigned<long, unsigned long>();
  test_make_unsigned<unsigned long, unsigned long>();
  test_make_unsigned<long long, unsigned long long>();
  test_make_unsigned<unsigned long long, unsigned long long>();
  test_make_unsigned<wchar_t, cuda::std::conditional<sizeof(wchar_t) == 4, unsigned int, unsigned short>::type>();
  test_make_unsigned<const wchar_t,
                     cuda::std::conditional<sizeof(wchar_t) == 4, const unsigned int, const unsigned short>::type>();
  test_make_unsigned<
    const Enum,
    cuda::std::conditional<sizeof(Enum) == sizeof(int), const unsigned int, const unsigned char>::type>();
  test_make_unsigned<BigEnum, cuda::std::conditional<sizeof(long) == 4, unsigned long long, unsigned long>::type>();
#if _CCCL_HAS_INT128()
  test_make_unsigned<__int128_t, __uint128_t>();
  test_make_unsigned<__uint128_t, __uint128_t>();
  test_make_unsigned<HugeEnum, __uint128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
