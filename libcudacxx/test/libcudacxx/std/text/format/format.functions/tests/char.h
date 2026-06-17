//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
template <class CharT, class... Args>
TEST_FUNC bool
check(cuda::std::basic_string_view<CharT> expected, cuda::std::basic_string_view<CharT> fmt, Args&&... args);
template <class CharT, class... Args>
TEST_FUNC bool check_exception(cuda::std::string_view what, cuda::std::basic_string_view<CharT> fmt, Args&&... args);

template <class CharT>
TEST_FUNC void test_char()
{
  // ***** Char type *****
  // *** align-fill & width ***
  assert(check(SV("answer is '*     '"), SV("answer is '{:6}'"), CharT('*')));
  assert(check(SV("answer is '     *'"), SV("answer is '{:>6}'"), CharT('*')));
  assert(check(SV("answer is '*     '"), SV("answer is '{:<6}'"), CharT('*')));
  assert(check(SV("answer is '  *   '"), SV("answer is '{:^6}'"), CharT('*')));

  assert(check(SV("answer is '*     '"), SV("answer is '{:6c}'"), CharT('*')));
  assert(check(SV("answer is '     *'"), SV("answer is '{:>6c}'"), CharT('*')));
  assert(check(SV("answer is '*     '"), SV("answer is '{:<6c}'"), CharT('*')));
  assert(check(SV("answer is '  *   '"), SV("answer is '{:^6c}'"), CharT('*')));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::::*'"), SV("answer is '{::>6}'"), CharT('*')));
  assert(check(SV("answer is '*:::::'"), SV("answer is '{::<6}'"), CharT('*')));
  assert(check(SV("answer is '::*:::'"), SV("answer is '{::^6}'"), CharT('*')));

  assert(check(SV("answer is '-----*'"), SV("answer is '{:->6c}'"), CharT('*')));
  assert(check(SV("answer is '*-----'"), SV("answer is '{:-<6c}'"), CharT('*')));
  assert(check(SV("answer is '--*---'"), SV("answer is '{:-^6c}'"), CharT('*')));

  // *** Sign ***
  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{:-}"), CharT('*')));
  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{:+}"), CharT('*')));
  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{: }"), CharT('*')));

  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{:-c}"), CharT('*')));
  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{:+c}"), CharT('*')));
  assert(
    check_exception("The format specifier for a character does not allow the sign option", SV("{: c}"), CharT('*')));

  // *** alternate form ***
  assert(check_exception(
    "The format specifier for a character does not allow the alternate form option", SV("{:#}"), CharT('*')));
  assert(check_exception(
    "The format specifier for a character does not allow the alternate form option", SV("{:#c}"), CharT('*')));

  // *** zero-padding ***
  assert(check_exception(
    "The format specifier for a character does not allow the zero-padding option", SV("{:0}"), CharT('*')));
  assert(check_exception(
    "The format specifier for a character does not allow the zero-padding option", SV("{:0c}"), CharT('*')));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42}"), CharT('*')));

  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.c}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0c}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42c}"), CharT('*')));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // #if TEST_STD_VER > 20
  //   const char* valid_types = "bBcdoxX?";
  // #else
  //   const char* valid_types = "bBcdoxX";
  // #endif
  // for (const auto& fmt : invalid_types<CharT>(valid_types))
  // {
  //   check_exception("The type option contains an invalid value for a character formatting argument", fmt,
  //   CharT('*'));
  // }
}

template <class CharT>
TEST_FUNC void test_char_as_int()
{
  // *** align-fill & width ***
  assert(check(SV("answer is '42'"), SV("answer is '{:<1d}'"), CharT('*')));

  assert(check(SV("answer is '42'"), SV("answer is '{:<2d}'"), CharT('*')));
  assert(check(SV("answer is '42 '"), SV("answer is '{:<3d}'"), CharT('*')));

  assert(check(SV("answer is '     42'"), SV("answer is '{:7d}'"), CharT('*')));
  assert(check(SV("answer is '     42'"), SV("answer is '{:>7d}'"), CharT('*')));
  assert(check(SV("answer is '42     '"), SV("answer is '{:<7d}'"), CharT('*')));
  assert(check(SV("answer is '  42   '"), SV("answer is '{:^7d}'"), CharT('*')));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::::42'"), SV("answer is '{::>7d}'"), CharT('*')));
  assert(check(SV("answer is '42:::::'"), SV("answer is '{::<7d}'"), CharT('*')));
  assert(check(SV("answer is '::42:::'"), SV("answer is '{::^7d}'"), CharT('*')));

  // Test whether zero padding is ignored
  assert(check(SV("answer is '     42'"), SV("answer is '{:>07d}'"), CharT('*')));
  assert(check(SV("answer is '42     '"), SV("answer is '{:<07d}'"), CharT('*')));
  assert(check(SV("answer is '  42   '"), SV("answer is '{:^07d}'"), CharT('*')));

  // *** Sign ***
  assert(check(SV("answer is 42"), SV("answer is {:d}"), CharT('*')));
  assert(check(SV("answer is 42"), SV("answer is {:-d}"), CharT('*')));
  assert(check(SV("answer is +42"), SV("answer is {:+d}"), CharT('*')));
  assert(check(SV("answer is  42"), SV("answer is {: d}"), CharT('*')));

  // *** alternate form ***
  assert(check(SV("answer is +42"), SV("answer is {:+#d}"), CharT('*')));
  assert(check(SV("answer is +101010"), SV("answer is {:+b}"), CharT('*')));
  assert(check(SV("answer is +0b101010"), SV("answer is {:+#b}"), CharT('*')));
  assert(check(SV("answer is +0B101010"), SV("answer is {:+#B}"), CharT('*')));
  assert(check(SV("answer is +52"), SV("answer is {:+o}"), CharT('*')));
  assert(check(SV("answer is +052"), SV("answer is {:+#o}"), CharT('*')));
  assert(check(SV("answer is +2a"), SV("answer is {:+x}"), CharT('*')));
  assert(check(SV("answer is +0x2a"), SV("answer is {:+#x}"), CharT('*')));
  assert(check(SV("answer is +2A"), SV("answer is {:+X}"), CharT('*')));
  assert(check(SV("answer is +0X2A"), SV("answer is {:+#X}"), CharT('*')));

  // *** zero-padding & width ***
  assert(check(SV("answer is +00000000042"), SV("answer is {:+#012d}"), CharT('*')));
  assert(check(SV("answer is +00000101010"), SV("answer is {:+012b}"), CharT('*')));
  assert(check(SV("answer is +0b000101010"), SV("answer is {:+#012b}"), CharT('*')));
  assert(check(SV("answer is +0B000101010"), SV("answer is {:+#012B}"), CharT('*')));
  assert(check(SV("answer is +00000000052"), SV("answer is {:+012o}"), CharT('*')));
  assert(check(SV("answer is +00000000052"), SV("answer is {:+#012o}"), CharT('*')));
  assert(check(SV("answer is +0000000002a"), SV("answer is {:+012x}"), CharT('*')));
  assert(check(SV("answer is +0x00000002a"), SV("answer is {:+#012x}"), CharT('*')));
  assert(check(SV("answer is +0000000002A"), SV("answer is {:+012X}"), CharT('*')));

  assert(check(SV("answer is +0X00000002A"), SV("answer is {:+#012X}"), CharT('*')));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.d}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0d}"), CharT('*')));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42d}"), CharT('*')));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // #if TEST_STD_VER > 20
  //   const char* valid_types = "bBcdoxX?";
  // #else
  //   const char* valid_types = "bBcdoxX";
  // #endif
  // for (const auto& fmt : invalid_types<CharT>(valid_types))
  // {
  //   check_exception("The type option contains an invalid value for a character formatting argument", fmt,
  //   CharT('*'));
  // }
}

TEST_FUNC void test()
{
  test_char<char>();
  test_char_as_int<char>();
#if _CCCL_HAS_WCHAR_T()
  test_char<wchar_t>();
  test_char_as_int<wchar_t>();

  {
    using CharT = wchar_t;
    assert(check(SV("hello 09azA"), SV("hello {}{}{}{}{}"), '0', '9', 'a', 'z', 'A'));
  }
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
