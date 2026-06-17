//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
TEST_FUNC bool check(...);
TEST_FUNC bool check_exception(...);

template <class CharT, class I>
TEST_FUNC _CCCL_NOINLINE void test_integer_as_integer()
{
  // *** align-fill & width ***
  assert(check(SV("answer is '42'"), SV("answer is '{:<1}'"), I(42)));
  assert(check(SV("answer is '42'"), SV("answer is '{:<2}'"), I(42)));
  assert(check(SV("answer is '42 '"), SV("answer is '{:<3}'"), I(42)));

  assert(check(SV("answer is '     42'"), SV("answer is '{:7}'"), I(42)));
  assert(check(SV("answer is '     42'"), SV("answer is '{:>7}'"), I(42)));
  assert(check(SV("answer is '42     '"), SV("answer is '{:<7}'"), I(42)));
  assert(check(SV("answer is '  42   '"), SV("answer is '{:^7}'"), I(42)));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::::42'"), SV("answer is '{::>7}'"), I(42)));
  assert(check(SV("answer is '42:::::'"), SV("answer is '{::<7}'"), I(42)));
  assert(check(SV("answer is '::42:::'"), SV("answer is '{::^7}'"), I(42)));

  // Test whether zero padding is ignored
  assert(check(SV("answer is '     42'"), SV("answer is '{:>07}'"), I(42)));
  assert(check(SV("answer is '42     '"), SV("answer is '{:<07}'"), I(42)));
  assert(check(SV("answer is '  42   '"), SV("answer is '{:^07}'"), I(42)));

  // *** Sign ***
  assert(check(SV("answer is 0"), SV("answer is {}"), I(0)));
  assert(check(SV("answer is 42"), SV("answer is {}"), I(42)));

  assert(check(SV("answer is 0"), SV("answer is {:-}"), I(0)));
  assert(check(SV("answer is 42"), SV("answer is {:-}"), I(42)));

  assert(check(SV("answer is +0"), SV("answer is {:+}"), I(0)));
  assert(check(SV("answer is +42"), SV("answer is {:+}"), I(42)));

  assert(check(SV("answer is  0"), SV("answer is {: }"), I(0)));
  assert(check(SV("answer is  42"), SV("answer is {: }"), I(42)));

  // *** alternate form ***
  assert(check(SV("answer is 0"), SV("answer is {:#}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:#d}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:b}"), I(0)));
  assert(check(SV("answer is 0b0"), SV("answer is {:#b}"), I(0)));
  assert(check(SV("answer is 0B0"), SV("answer is {:#B}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:o}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:#o}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:x}"), I(0)));
  assert(check(SV("answer is 0x0"), SV("answer is {:#x}"), I(0)));
  assert(check(SV("answer is 0"), SV("answer is {:X}"), I(0)));
  assert(check(SV("answer is 0X0"), SV("answer is {:#X}"), I(0)));

  assert(check(SV("answer is +42"), SV("answer is {:+#}"), I(42)));
  assert(check(SV("answer is +42"), SV("answer is {:+#d}"), I(42)));
  assert(check(SV("answer is +101010"), SV("answer is {:+b}"), I(42)));
  assert(check(SV("answer is +0b101010"), SV("answer is {:+#b}"), I(42)));
  assert(check(SV("answer is +0B101010"), SV("answer is {:+#B}"), I(42)));
  assert(check(SV("answer is +52"), SV("answer is {:+o}"), I(42)));
  assert(check(SV("answer is +052"), SV("answer is {:+#o}"), I(42)));
  assert(check(SV("answer is +2a"), SV("answer is {:+x}"), I(42)));
  assert(check(SV("answer is +0x2a"), SV("answer is {:+#x}"), I(42)));
  assert(check(SV("answer is +2A"), SV("answer is {:+X}"), I(42)));
  assert(check(SV("answer is +0X2A"), SV("answer is {:+#X}"), I(42)));

  // *** zero-padding & width ***
  assert(check(SV("answer is 000000000000"), SV("answer is {:#012}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:#012d}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012b}"), I(0)));
  assert(check(SV("answer is 0b0000000000"), SV("answer is {:#012b}"), I(0)));
  assert(check(SV("answer is 0B0000000000"), SV("answer is {:#012B}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012o}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:#012o}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012x}"), I(0)));
  assert(check(SV("answer is 0x0000000000"), SV("answer is {:#012x}"), I(0)));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012X}"), I(0)));
  assert(check(SV("answer is 0X0000000000"), SV("answer is {:#012X}"), I(0)));

  assert(check(SV("answer is +00000000042"), SV("answer is {:+#012}"), I(42)));
  assert(check(SV("answer is +00000000042"), SV("answer is {:+#012d}"), I(42)));
  assert(check(SV("answer is +00000101010"), SV("answer is {:+012b}"), I(42)));
  assert(check(SV("answer is +0b000101010"), SV("answer is {:+#012b}"), I(42)));
  assert(check(SV("answer is +0B000101010"), SV("answer is {:+#012B}"), I(42)));
  assert(check(SV("answer is +00000000052"), SV("answer is {:+012o}"), I(42)));
  assert(check(SV("answer is +00000000052"), SV("answer is {:+#012o}"), I(42)));
  assert(check(SV("answer is +0000000002a"), SV("answer is {:+012x}"), I(42)));
  assert(check(SV("answer is +0x00000002a"), SV("answer is {:+#012x}"), I(42)));
  assert(check(SV("answer is +0000000002A"), SV("answer is {:+012X}"), I(42)));
  assert(check(SV("answer is +0X00000002A"), SV("answer is {:+#012X}"), I(42)));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42}"), I(0)));

  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), I(0), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), I(0), 1.0));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
  // {
  //   check_exception("The type option contains an invalid value for an integer formatting argument", fmt, I(0));
  // }
}

template <class CharT, class I>
TEST_FUNC _CCCL_NOINLINE void test_integer_as_char()
{
  // *** align-fill & width ***
  assert(check(SV("answer is '*     '"), SV("answer is '{:6c}'"), I(42)));
  assert(check(SV("answer is '     *'"), SV("answer is '{:>6c}'"), I(42)));
  assert(check(SV("answer is '*     '"), SV("answer is '{:<6c}'"), I(42)));
  assert(check(SV("answer is '  *   '"), SV("answer is '{:^6c}'"), I(42)));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::::*'"), SV("answer is '{::>6c}'"), I(42)));
  assert(check(SV("answer is '*:::::'"), SV("answer is '{::<6c}'"), I(42)));
  assert(check(SV("answer is '::*:::'"), SV("answer is '{::^6c}'"), I(42)));

  // *** Sign ***
  assert(check(SV("answer is *"), SV("answer is {:c}"), I(42)));
  assert(check_exception(
    "The format specifier for an integer does not allow the sign option", SV("answer is {:-c}"), I(42)));
  assert(check_exception(
    "The format specifier for an integer does not allow the sign option", SV("answer is {:+c}"), I(42)));
  assert(check_exception(
    "The format specifier for an integer does not allow the sign option", SV("answer is {: c}"), I(42)));

  // *** alternate form ***
  assert(check_exception(
    "The format specifier for an integer does not allow the alternate form option", SV("answer is {:#c}"), I(42)));

  // *** zero-padding & width ***
  assert(check_exception(
    "The format specifier for an integer does not allow the zero-padding option", SV("answer is {:01c}"), I(42)));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.c}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0c}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42c}"), I(0)));

  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}c}"), I(0)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}c}"), I(0), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}c}"), I(0), 1.0));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("bBcdoxX"))
  // {
  //   check_exception("The type option contains an invalid value for an integer formatting argument", fmt, I(42));
  // }

  // *** Validate range ***
  // The code has some duplications to keep the if statement readable.
  if constexpr (cuda::std::signed_integral<CharT>)
  {
    if constexpr (sizeof(I) >= sizeof(CharT))
    {
      assert(check_exception(
        "Integral value outside the range of the char type", SV("{:c}"), cuda::std::numeric_limits<I>::max()));
    }
  }
  else if constexpr (sizeof(I) > sizeof(CharT))
  {
    assert(check_exception(
      "Integral value outside the range of the char type", SV("{:c}"), cuda::std::numeric_limits<I>::max()));
  }
}

template <class CharT, class T>
TEST_FUNC void test_integer()
{
  test_integer_as_integer<CharT, T>();
  test_integer_as_char<CharT, T>();
}

template <class CharT>
TEST_FUNC _CCCL_NOINLINE void test_max()
{
  // *** test the maxima ***
  assert(check(SV("0b11111111"), SV("{:#b}"), cuda::std::numeric_limits<cuda::std::uint8_t>::max()));
  assert(check(SV("0377"), SV("{:#o}"), cuda::std::numeric_limits<cuda::std::uint8_t>::max()));
  assert(check(SV("255"), SV("{:#}"), cuda::std::numeric_limits<cuda::std::uint8_t>::max()));
  assert(check(SV("0xff"), SV("{:#x}"), cuda::std::numeric_limits<cuda::std::uint8_t>::max()));

  assert(check(SV("0b1111111111111111"), SV("{:#b}"), cuda::std::numeric_limits<cuda::std::uint16_t>::max()));
  assert(check(SV("0177777"), SV("{:#o}"), cuda::std::numeric_limits<cuda::std::uint16_t>::max()));
  assert(check(SV("65535"), SV("{:#}"), cuda::std::numeric_limits<cuda::std::uint16_t>::max()));
  assert(check(SV("0xffff"), SV("{:#x}"), cuda::std::numeric_limits<cuda::std::uint16_t>::max()));

  assert(check(
    SV("0b11111111111111111111111111111111"), SV("{:#b}"), cuda::std::numeric_limits<cuda::std::uint32_t>::max()));
  assert(check(SV("037777777777"), SV("{:#o}"), cuda::std::numeric_limits<cuda::std::uint32_t>::max()));
  assert(check(SV("4294967295"), SV("{:#}"), cuda::std::numeric_limits<cuda::std::uint32_t>::max()));
  assert(check(SV("0xffffffff"), SV("{:#x}"), cuda::std::numeric_limits<cuda::std::uint32_t>::max()));

  assert(check(SV("0b1111111111111111111111111111111111111111111111111111111111111111"),
               SV("{:#b}"),
               cuda::std::numeric_limits<cuda::std::uint64_t>::max()));
  assert(check(SV("01777777777777777777777"), SV("{:#o}"), cuda::std::numeric_limits<cuda::std::uint64_t>::max()));
  assert(check(SV("18446744073709551615"), SV("{:#}"), cuda::std::numeric_limits<cuda::std::uint64_t>::max()));
  assert(check(SV("0xffffffffffffffff"), SV("{:#x}"), cuda::std::numeric_limits<cuda::std::uint64_t>::max()));

#if _CCCL_HAS_INT128()
  assert(check(SV("0b1111111111111111111111111111111111111111111111111111111111111111"
                  "1111111111111111111111111111111111111111111111111111111111111111"),
               SV("{:#b}"),
               cuda::std::numeric_limits<__uint128_t>::max()));
  assert(check(
    SV("03777777777777777777777777777777777777777777"), SV("{:#o}"), cuda::std::numeric_limits<__uint128_t>::max()));
  assert(
    check(SV("340282366920938463463374607431768211455"), SV("{:#}"), cuda::std::numeric_limits<__uint128_t>::max()));
  assert(check(SV("0xffffffffffffffffffffffffffffffff"), SV("{:#x}"), cuda::std::numeric_limits<__uint128_t>::max()));
#endif // _CCCL_HAS_INT128()
}

template <class CharT>
TEST_FUNC _CCCL_NOINLINE void test()
{
  test_integer<CharT, unsigned char>();
  test_integer<CharT, unsigned short>();
  test_integer<CharT, unsigned int>();
  test_integer<CharT, unsigned long>();
  test_integer<CharT, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_integer<CharT, __uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_max<CharT>();
}

TEST_FUNC void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
