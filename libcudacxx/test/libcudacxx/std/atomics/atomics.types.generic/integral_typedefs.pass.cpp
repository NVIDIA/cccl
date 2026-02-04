//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// using atomic_char      = atomic<char>;
// using atomic_schar     = atomic<signed char>;
// using atomic_uchar     = atomic<unsigned char>;
// using atomic_short     = atomic<short>;
// using atomic_ushort    = atomic<unsigned short>;
// using atomic_int       = atomic<int>;
// using atomic_uint      = atomic<unsigned int>;
// using atomic_long      = atomic<long>;
// using atomic_ulong     = atomic<unsigned long>;
// using atomic_llong     = atomic<long long>;
// using atomic_ullong    = atomic<unsigned long long>;
// using atomic_char16_t  = atomic<char16_t>;
// using atomic_char32_t  = atomic<char32_t>;
// using atomic_wchar_t   = atomic<wchar_t>;
//
// using atomic_intptr_t  = atomic<intptr_t>;
// using atomic_uintptr_t = atomic<uintptr_t>;
//
// using atomic_int8_t    = atomic<int8_t>;
// using atomic_uint8_t   = atomic<uint8_t>;
// using atomic_int16_t   = atomic<int16_t>;
// using atomic_uint16_t  = atomic<uint16_t>;
// using atomic_int32_t   = atomic<int32_t>;
// using atomic_uint32_t  = atomic<uint32_t>;
// using atomic_int64_t   = atomic<int64_t>;
// using atomic_uint64_t  = atomic<uint64_t>;

#include <cuda/std/atomic>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::atomic<char>, cuda::std::atomic_char>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<signed char>, cuda::std::atomic_schar>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<unsigned char>, cuda::std::atomic_uchar>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<short>, cuda::std::atomic_short>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<unsigned short>, cuda::std::atomic_ushort>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<int>, cuda::std::atomic_int>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<unsigned int>, cuda::std::atomic_uint>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<long>, cuda::std::atomic_long>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<unsigned long>, cuda::std::atomic_ulong>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<long long>, cuda::std::atomic_llong>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<unsigned long long>, cuda::std::atomic_ullong>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<wchar_t>, cuda::std::atomic_wchar_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<char16_t>, cuda::std::atomic_char16_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<char32_t>, cuda::std::atomic_char32_t>::value), "");

  //  Added by LWG 2441
  static_assert((cuda::std::is_same<cuda::std::atomic<intptr_t>, cuda::std::atomic_intptr_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<uintptr_t>, cuda::std::atomic_uintptr_t>::value), "");

  static_assert((cuda::std::is_same<cuda::std::atomic<int8_t>, cuda::std::atomic_int8_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<uint8_t>, cuda::std::atomic_uint8_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<int16_t>, cuda::std::atomic_int16_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<uint16_t>, cuda::std::atomic_uint16_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<int32_t>, cuda::std::atomic_int32_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<uint32_t>, cuda::std::atomic_uint32_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<int64_t>, cuda::std::atomic_int64_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<uint64_t>, cuda::std::atomic_uint64_t>::value), "");

  return 0;
}
