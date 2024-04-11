//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <cuda/std/cstdint>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
// #include <cuda/std/cwchar>
// #include <cuda/std/csignal>
// #include <cuda/std/cwctype>
#include <cuda/std/climits>
#include <cuda/std/type_traits>
// #include <cuda/std/limits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  // typedef cuda::std::int8_t
  static_assert(sizeof(cuda::std::int8_t) * CHAR_BIT == 8, "sizeof(cuda::std::int8_t)*CHAR_BIT == 8");
  static_assert(cuda::std::is_signed<cuda::std::int8_t>::value, "cuda::std::is_signed<cuda::std::int8_t>::value");
  // typedef cuda::std::int16_t
  static_assert(sizeof(cuda::std::int16_t) * CHAR_BIT == 16, "sizeof(cuda::std::int16_t)*CHAR_BIT == 16");
  static_assert(cuda::std::is_signed<cuda::std::int16_t>::value, "cuda::std::is_signed<cuda::std::int16_t>::value");
  // typedef cuda::std::int32_t
  static_assert(sizeof(cuda::std::int32_t) * CHAR_BIT == 32, "sizeof(cuda::std::int32_t)*CHAR_BIT == 32");
  static_assert(cuda::std::is_signed<cuda::std::int32_t>::value, "cuda::std::is_signed<cuda::std::int32_t>::value");
  // typedef cuda::std::int64_t
  static_assert(sizeof(cuda::std::int64_t) * CHAR_BIT == 64, "sizeof(cuda::std::int64_t)*CHAR_BIT == 64");
  static_assert(cuda::std::is_signed<cuda::std::int64_t>::value, "cuda::std::is_signed<cuda::std::int64_t>::value");

  // typedef cuda::std::uint8_t
  static_assert(sizeof(cuda::std::uint8_t) * CHAR_BIT == 8, "sizeof(cuda::std::uint8_t)*CHAR_BIT == 8");
  static_assert(cuda::std::is_unsigned<cuda::std::uint8_t>::value, "cuda::std::is_unsigned<cuda::std::uint8_t>::value");
  // typedef cuda::std::uint16_t
  static_assert(sizeof(cuda::std::uint16_t) * CHAR_BIT == 16, "sizeof(cuda::std::uint16_t)*CHAR_BIT == 16");
  static_assert(cuda::std::is_unsigned<cuda::std::uint16_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint16_t>::value");
  // typedef cuda::std::uint32_t
  static_assert(sizeof(cuda::std::uint32_t) * CHAR_BIT == 32, "sizeof(cuda::std::uint32_t)*CHAR_BIT == 32");
  static_assert(cuda::std::is_unsigned<cuda::std::uint32_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint32_t>::value");
  // typedef cuda::std::uint64_t
  static_assert(sizeof(cuda::std::uint64_t) * CHAR_BIT == 64, "sizeof(cuda::std::uint64_t)*CHAR_BIT == 64");
  static_assert(cuda::std::is_unsigned<cuda::std::uint64_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint64_t>::value");

  // typedef cuda::std::int_least8_t
  static_assert(sizeof(cuda::std::int_least8_t) * CHAR_BIT >= 8, "sizeof(cuda::std::int_least8_t)*CHAR_BIT >= 8");
  static_assert(cuda::std::is_signed<cuda::std::int_least8_t>::value,
                "cuda::std::is_signed<cuda::std::int_least8_t>::value");
  // typedef cuda::std::int_least16_t
  static_assert(sizeof(cuda::std::int_least16_t) * CHAR_BIT >= 16, "sizeof(cuda::std::int_least16_t)*CHAR_BIT >= 16");
  static_assert(cuda::std::is_signed<cuda::std::int_least16_t>::value,
                "cuda::std::is_signed<cuda::std::int_least16_t>::value");
  // typedef cuda::std::int_least32_t
  static_assert(sizeof(cuda::std::int_least32_t) * CHAR_BIT >= 32, "sizeof(cuda::std::int_least32_t)*CHAR_BIT >= 32");
  static_assert(cuda::std::is_signed<cuda::std::int_least32_t>::value,
                "cuda::std::is_signed<cuda::std::int_least32_t>::value");
  // typedef cuda::std::int_least64_t
  static_assert(sizeof(cuda::std::int_least64_t) * CHAR_BIT >= 64, "sizeof(cuda::std::int_least64_t)*CHAR_BIT >= 64");
  static_assert(cuda::std::is_signed<cuda::std::int_least64_t>::value,
                "cuda::std::is_signed<cuda::std::int_least64_t>::value");

  // typedef cuda::std::uint_least8_t
  static_assert(sizeof(cuda::std::uint_least8_t) * CHAR_BIT >= 8, "sizeof(cuda::std::uint_least8_t)*CHAR_BIT >= 8");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_least8_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_least8_t>::value");
  // typedef cuda::std::uint_least16_t
  static_assert(sizeof(cuda::std::uint_least16_t) * CHAR_BIT >= 16, "sizeof(cuda::std::uint_least16_t)*CHAR_BIT >= 16");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_least16_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_least16_t>::value");
  // typedef cuda::std::uint_least32_t
  static_assert(sizeof(cuda::std::uint_least32_t) * CHAR_BIT >= 32, "sizeof(cuda::std::uint_least32_t)*CHAR_BIT >= 32");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_least32_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_least32_t>::value");
  // typedef cuda::std::uint_least64_t
  static_assert(sizeof(cuda::std::uint_least64_t) * CHAR_BIT >= 64, "sizeof(cuda::std::uint_least64_t)*CHAR_BIT >= 64");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_least64_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_least64_t>::value");

  // typedef cuda::std::int_fast8_t
  static_assert(sizeof(cuda::std::int_fast8_t) * CHAR_BIT >= 8, "sizeof(cuda::std::int_fast8_t)*CHAR_BIT >= 8");
  static_assert(cuda::std::is_signed<cuda::std::int_fast8_t>::value,
                "cuda::std::is_signed<cuda::std::int_fast8_t>::value");
  // typedef cuda::std::int_fast16_t
  static_assert(sizeof(cuda::std::int_fast16_t) * CHAR_BIT >= 16, "sizeof(cuda::std::int_fast16_t)*CHAR_BIT >= 16");
  static_assert(cuda::std::is_signed<cuda::std::int_fast16_t>::value,
                "cuda::std::is_signed<cuda::std::int_fast16_t>::value");
  // typedef cuda::std::int_fast32_t
  static_assert(sizeof(cuda::std::int_fast32_t) * CHAR_BIT >= 32, "sizeof(cuda::std::int_fast32_t)*CHAR_BIT >= 32");
  static_assert(cuda::std::is_signed<cuda::std::int_fast32_t>::value,
                "cuda::std::is_signed<cuda::std::int_fast32_t>::value");
  // typedef cuda::std::int_fast64_t
  static_assert(sizeof(cuda::std::int_fast64_t) * CHAR_BIT >= 64, "sizeof(cuda::std::int_fast64_t)*CHAR_BIT >= 64");
  static_assert(cuda::std::is_signed<cuda::std::int_fast64_t>::value,
                "cuda::std::is_signed<cuda::std::int_fast64_t>::value");

  // typedef cuda::std::uint_fast8_t
  static_assert(sizeof(cuda::std::uint_fast8_t) * CHAR_BIT >= 8, "sizeof(cuda::std::uint_fast8_t)*CHAR_BIT >= 8");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_fast8_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_fast8_t>::value");
  // typedef cuda::std::uint_fast16_t
  static_assert(sizeof(cuda::std::uint_fast16_t) * CHAR_BIT >= 16, "sizeof(cuda::std::uint_fast16_t)*CHAR_BIT >= 16");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_fast16_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_fast16_t>::value");
  // typedef cuda::std::uint_fast32_t
  static_assert(sizeof(cuda::std::uint_fast32_t) * CHAR_BIT >= 32, "sizeof(cuda::std::uint_fast32_t)*CHAR_BIT >= 32");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_fast32_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_fast32_t>::value");
  // typedef cuda::std::uint_fast64_t
  static_assert(sizeof(cuda::std::uint_fast64_t) * CHAR_BIT >= 64, "sizeof(cuda::std::uint_fast64_t)*CHAR_BIT >= 64");
  static_assert(cuda::std::is_unsigned<cuda::std::uint_fast64_t>::value,
                "cuda::std::is_unsigned<cuda::std::uint_fast64_t>::value");

  // typedef cuda::std::intptr_t
  static_assert(sizeof(cuda::std::intptr_t) >= sizeof(void*), "sizeof(cuda::std::intptr_t) >= sizeof(void*)");
  static_assert(cuda::std::is_signed<cuda::std::intptr_t>::value, "cuda::std::is_signed<cuda::std::intptr_t>::value");
  // typedef cuda::std::uintptr_t
  static_assert(sizeof(cuda::std::uintptr_t) >= sizeof(void*), "sizeof(cuda::std::uintptr_t) >= sizeof(void*)");
  static_assert(cuda::std::is_unsigned<cuda::std::uintptr_t>::value,
                "cuda::std::is_unsigned<cuda::std::uintptr_t>::value");

  // typedef cuda::std::intmax_t
  static_assert(sizeof(cuda::std::intmax_t) >= sizeof(long long), "sizeof(cuda::std::intmax_t) >= sizeof(long long)");
  static_assert(cuda::std::is_signed<cuda::std::intmax_t>::value, "cuda::std::is_signed<cuda::std::intmax_t>::value");
  // typedef cuda::std::uintmax_t
  static_assert(sizeof(cuda::std::uintmax_t) >= sizeof(unsigned long long),
                "sizeof(cuda::std::uintmax_t) >= sizeof(unsigned long long)");
  static_assert(cuda::std::is_unsigned<cuda::std::uintmax_t>::value,
                "cuda::std::is_unsigned<cuda::std::uintmax_t>::value");

  // INTN_MIN
  static_assert(INT8_MIN == -128, "INT8_MIN == -128");
  static_assert(INT16_MIN == -32768, "INT16_MIN == -32768");
  static_assert(INT32_MIN == -2147483647 - 1, "INT32_MIN == -2147483648");
  static_assert(INT64_MIN == -9223372036854775807LL - 1, "INT64_MIN == -9223372036854775808LL");

  // INTN_MAX
  static_assert(INT8_MAX == 127, "INT8_MAX == 127");
  static_assert(INT16_MAX == 32767, "INT16_MAX == 32767");
  static_assert(INT32_MAX == 2147483647, "INT32_MAX == 2147483647");
  static_assert(INT64_MAX == 9223372036854775807LL, "INT64_MAX == 9223372036854775807LL");

  // UINTN_MAX
  static_assert(UINT8_MAX == 255, "UINT8_MAX == 255");
  static_assert(UINT16_MAX == 65535, "UINT16_MAX == 65535");
  static_assert(UINT32_MAX == 4294967295U, "UINT32_MAX == 4294967295");
  static_assert(UINT64_MAX == 18446744073709551615ULL, "UINT64_MAX == 18446744073709551615ULL");

  // INT_FASTN_MIN
  static_assert(INT_FAST8_MIN <= -128, "INT_FAST8_MIN <= -128");
  static_assert(INT_FAST16_MIN <= -32768, "INT_FAST16_MIN <= -32768");
  static_assert(INT_FAST32_MIN <= -2147483647 - 1, "INT_FAST32_MIN <= -2147483648");
  static_assert(INT_FAST64_MIN <= -9223372036854775807LL - 1, "INT_FAST64_MIN <= -9223372036854775808LL");

  // INT_FASTN_MAX
  static_assert(INT_FAST8_MAX >= 127, "INT_FAST8_MAX >= 127");
  static_assert(INT_FAST16_MAX >= 32767, "INT_FAST16_MAX >= 32767");
  static_assert(INT_FAST32_MAX >= 2147483647, "INT_FAST32_MAX >= 2147483647");
  static_assert(INT_FAST64_MAX >= 9223372036854775807LL, "INT_FAST64_MAX >= 9223372036854775807LL");

  // UINT_FASTN_MAX
  static_assert(UINT_FAST8_MAX >= 255, "UINT_FAST8_MAX >= 255");
  static_assert(UINT_FAST16_MAX >= 65535, "UINT_FAST16_MAX >= 65535");
  static_assert(UINT_FAST32_MAX >= 4294967295U, "UINT_FAST32_MAX >= 4294967295");
  static_assert(UINT_FAST64_MAX >= 18446744073709551615ULL, "UINT_FAST64_MAX >= 18446744073709551615ULL");

#if 0
    // INTPTR_MIN
    assert(INTPTR_MIN == cuda::std::numeric_limits<cuda::std::intptr_t>::min());

    // INTPTR_MAX
    assert(INTPTR_MAX == cuda::std::numeric_limits<cuda::std::intptr_t>::max());

    // UINTPTR_MAX
    assert(UINTPTR_MAX == cuda::std::numeric_limits<cuda::std::uintptr_t>::max());

    // INTMAX_MIN
    assert(INTMAX_MIN == cuda::std::numeric_limits<cuda::std::intmax_t>::min());

    // INTMAX_MAX
    assert(INTMAX_MAX == cuda::std::numeric_limits<cuda::std::intmax_t>::max());

    // UINTMAX_MAX
    assert(UINTMAX_MAX == cuda::std::numeric_limits<cuda::std::uintmax_t>::max());

    // PTRDIFF_MIN
    assert(PTRDIFF_MIN == cuda::std::numeric_limits<cuda::std::ptrdiff_t>::min());

    // PTRDIFF_MAX
    assert(PTRDIFF_MAX == cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max());

    // SIG_ATOMIC_MIN
    // assert(SIG_ATOMIC_MIN == cuda::std::numeric_limits<cuda::std::sig_atomic_t>::min());

    // SIG_ATOMIC_MAX
    // assert(SIG_ATOMIC_MAX == cuda::std::numeric_limits<cuda::std::sig_atomic_t>::max());

    // SIZE_MAX
    assert(SIZE_MAX == cuda::std::numeric_limits<cuda::std::size_t>::max());

    // WCHAR_MIN
    // assert(WCHAR_MIN == cuda::std::numeric_limits<wchar_t>::min());

    // WCHAR_MAX
    // assert(WCHAR_MAX == cuda::std::numeric_limits<wchar_t>::max());

    // WINT_MIN
    // assert(WINT_MIN == cuda::std::numeric_limits<cuda::std::wint_t>::min());

    // WINT_MAX
    // assert(WINT_MAX == cuda::std::numeric_limits<cuda::std::wint_t>::max());
#endif

#ifndef INT8_C
#  error INT8_C not defined
#endif

#ifndef INT16_C
#  error INT16_C not defined
#endif

#ifndef INT32_C
#  error INT32_C not defined
#endif

#ifndef INT64_C
#  error INT64_C not defined
#endif

#ifndef UINT8_C
#  error UINT8_C not defined
#endif

#ifndef UINT16_C
#  error UINT16_C not defined
#endif

#ifndef UINT32_C
#  error UINT32_C not defined
#endif

#ifndef UINT64_C
#  error UINT64_C not defined
#endif

#ifndef INTMAX_C
#  error INTMAX_C not defined
#endif

#ifndef UINTMAX_C
#  error UINTMAX_C not defined
#endif

  return 0;
}
