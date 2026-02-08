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

// using atomic_int_least8_t   = atomic<int_least8_t>;
// using atomic_uint_least8_t  = atomic<uint_least8_t>;
// using atomic_int_least16_t  = atomic<int_least16_t>;
// using atomic_uint_least16_t = atomic<uint_least16_t>;
// using atomic_int_least32_t  = atomic<int_least32_t>;
// using atomic_uint_least32_t = atomic<uint_least32_t>;
// using atomic_int_least64_t  = atomic<int_least64_t>;
// using atomic_uint_least64_t = atomic<uint_least64_t>;
//
// using atomic_int_fast8_t    = atomic<int_fast8_t>;
// using atomic_uint_fast8_t   = atomic<uint_fast8_t>;
// using atomic_int_fast16_t   = atomic<int_fast16_t>;
// using atomic_uint_fast16_t  = atomic<uint_fast16_t>;
// using atomic_int_fast32_t   = atomic<int_fast32_t>;
// using atomic_uint_fast32_t  = atomic<uint_fast32_t>;
// using atomic_int_fast64_t   = atomic<int_fast64_t>;
// using atomic_uint_fast64_t  = atomic<uint_fast64_t>;
//
// using atomic_intptr_t       = atomic<intptr_t>;
// using atomic_uintptr_t      = atomic<uintptr_t>;
// using atomic_size_t         = atomic<size_t>;
// using atomic_ptrdiff_t      = atomic<ptrdiff_t>;
// using atomic_intmax_t       = atomic<intmax_t>;
// using atomic_uintmax_t      = atomic<uintmax_t>;

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::int_least8_t>, cuda::std::atomic_int_least8_t>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_least8_t>, cuda::std::atomic_uint_least8_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::int_least16_t>, cuda::std::atomic_int_least16_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_least16_t>, cuda::std::atomic_uint_least16_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::int_least32_t>, cuda::std::atomic_int_least32_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_least32_t>, cuda::std::atomic_uint_least32_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::int_least64_t>, cuda::std::atomic_int_least64_t>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_least64_t>, cuda::std::atomic_uint_least64_t>::value), "");

  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::int_fast8_t>, cuda::std::atomic_int_fast8_t>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::uint_fast8_t>, cuda::std::atomic_uint_fast8_t>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::int_fast16_t>, cuda::std::atomic_int_fast16_t>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_fast16_t>, cuda::std::atomic_uint_fast16_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::int_fast32_t>, cuda::std::atomic_int_fast32_t>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_fast32_t>, cuda::std::atomic_uint_fast32_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::int_fast64_t>, cuda::std::atomic_int_fast64_t>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::atomic<cuda::std::uint_fast64_t>, cuda::std::atomic_uint_fast64_t>::value), "");

  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::intptr_t>, cuda::std::atomic_intptr_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::uintptr_t>, cuda::std::atomic_uintptr_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::size_t>, cuda::std::atomic_size_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::ptrdiff_t>, cuda::std::atomic_ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::intmax_t>, cuda::std::atomic_intmax_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::atomic<cuda::std::uintmax_t>, cuda::std::atomic_uintmax_t>::value), "");

  return 0;
}
