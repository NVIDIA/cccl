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
TEST_FUNC bool check(...);
TEST_FUNC bool check_exception(...);

template <class CharT>
TEST_FUNC void test_bool()
{
  // *** align-fill & width ***
  assert(check(SV("answer is 'true   '"), SV("answer is '{:7}'"), true));
  assert(check(SV("answer is '   true'"), SV("answer is '{:>7}'"), true));
  assert(check(SV("answer is 'true   '"), SV("answer is '{:<7}'"), true));
  assert(check(SV("answer is ' true  '"), SV("answer is '{:^7}'"), true));

  assert(check(SV("answer is 'false   '"), SV("answer is '{:8s}'"), false));
  assert(check(SV("answer is '   false'"), SV("answer is '{:>8s}'"), false));
  assert(check(SV("answer is 'false   '"), SV("answer is '{:<8s}'"), false));
  assert(check(SV("answer is ' false  '"), SV("answer is '{:^8s}'"), false));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::true'"), SV("answer is '{::>7}'"), true));
  assert(check(SV("answer is 'true:::'"), SV("answer is '{::<7}'"), true));
  assert(check(SV("answer is ':true::'"), SV("answer is '{::^7}'"), true));

  assert(check(SV("answer is '---false'"), SV("answer is '{:->8s}'"), false));
  assert(check(SV("answer is 'false---'"), SV("answer is '{:-<8s}'"), false));
  assert(check(SV("answer is '-false--'"), SV("answer is '{:-^8s}'"), false));

  // *** Sign ***
  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{:-}"), true));
  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{:+}"), true));
  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{: }"), true));

  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{:-s}"), true));
  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{:+s}"), true));
  assert(check_exception("The format specifier for a bool does not allow the sign option", SV("{: s}"), true));

  // *** alternate form ***
  assert(check_exception("The format specifier for a bool does not allow the alternate form option", SV("{:#}"), true));
  assert(
    check_exception("The format specifier for a bool does not allow the alternate form option", SV("{:#s}"), true));

  // *** zero-padding ***
  assert(check_exception("The format specifier for a bool does not allow the zero-padding option", SV("{:0}"), true));
  assert(check_exception("The format specifier for a bool does not allow the zero-padding option", SV("{:0s}"), true));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42}"), true));

  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.s}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0s}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42s}"), true));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("bBdosxX"))
  // {
  //   assert(check_exception("The type option contains an invalid value for a bool formatting argument", fmt, true));
  // }
}

template <class CharT>
TEST_FUNC void test_bool_as_integer()
{
  // *** align-fill & width ***
  assert(check(SV("answer is '1'"), SV("answer is '{:<1d}'"), true));
  assert(check(SV("answer is '1 '"), SV("answer is '{:<2d}'"), true));
  assert(check(SV("answer is '0 '"), SV("answer is '{:<2d}'"), false));

  assert(check(SV("answer is '     1'"), SV("answer is '{:6d}'"), true));
  assert(check(SV("answer is '     1'"), SV("answer is '{:>6d}'"), true));
  assert(check(SV("answer is '1     '"), SV("answer is '{:<6d}'"), true));
  assert(check(SV("answer is '  1   '"), SV("answer is '{:^6d}'"), true));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::::0'"), SV("answer is '{::>6d}'"), false));
  assert(check(SV("answer is '0:::::'"), SV("answer is '{::<6d}'"), false));
  assert(check(SV("answer is '::0:::'"), SV("answer is '{::^6d}'"), false));

  // Test whether zero padding is ignored
  assert(check(SV("answer is '     1'"), SV("answer is '{:>06d}'"), true));
  assert(check(SV("answer is '1     '"), SV("answer is '{:<06d}'"), true));
  assert(check(SV("answer is '  1   '"), SV("answer is '{:^06d}'"), true));

  // *** Sign ***
  assert(check(SV("answer is 1"), SV("answer is {:d}"), true));
  assert(check(SV("answer is 0"), SV("answer is {:-d}"), false));
  assert(check(SV("answer is +1"), SV("answer is {:+d}"), true));
  assert(check(SV("answer is  0"), SV("answer is {: d}"), false));

  // *** alternate form ***
  assert(check(SV("answer is +1"), SV("answer is {:+#d}"), true));
  assert(check(SV("answer is +1"), SV("answer is {:+b}"), true));
  assert(check(SV("answer is +0b1"), SV("answer is {:+#b}"), true));
  assert(check(SV("answer is +0B1"), SV("answer is {:+#B}"), true));
  assert(check(SV("answer is +1"), SV("answer is {:+o}"), true));
  assert(check(SV("answer is +01"), SV("answer is {:+#o}"), true));
  assert(check(SV("answer is +1"), SV("answer is {:+x}"), true));
  assert(check(SV("answer is +0x1"), SV("answer is {:+#x}"), true));
  assert(check(SV("answer is +1"), SV("answer is {:+X}"), true));
  assert(check(SV("answer is +0X1"), SV("answer is {:+#X}"), true));

  assert(check(SV("answer is 0"), SV("answer is {:#d}"), false));
  assert(check(SV("answer is 0"), SV("answer is {:b}"), false));
  assert(check(SV("answer is 0b0"), SV("answer is {:#b}"), false));
  assert(check(SV("answer is 0B0"), SV("answer is {:#B}"), false));
  assert(check(SV("answer is 0"), SV("answer is {:o}"), false));
  assert(check(SV("answer is 0"), SV("answer is {:#o}"), false));
  assert(check(SV("answer is 0"), SV("answer is {:x}"), false));
  assert(check(SV("answer is 0x0"), SV("answer is {:#x}"), false));
  assert(check(SV("answer is 0"), SV("answer is {:X}"), false));
  assert(check(SV("answer is 0X0"), SV("answer is {:#X}"), false));

  // todo(dabayer): Make tests that are commented out work.
  // *** zero-padding & width ***
  assert(check(SV("answer is +00000000001"), SV("answer is {:+#012d}"), true));
  assert(check(SV("answer is +00000000001"), SV("answer is {:+012b}"), true));
  assert(check(SV("answer is +0b000000001"), SV("answer is {:+#012b}"), true));
  assert(check(SV("answer is +0B000000001"), SV("answer is {:+#012B}"), true));
  assert(check(SV("answer is +00000000001"), SV("answer is {:+012o}"), true));
  assert(check(SV("answer is +00000000001"), SV("answer is {:+#012o}"), true));
  assert(check(SV("answer is +00000000001"), SV("answer is {:+012x}"), true));
  assert(check(SV("answer is +0x000000001"), SV("answer is {:+#012x}"), true));
  assert(check(SV("answer is +00000000001"), SV("answer is {:+012X}"), true));
  assert(check(SV("answer is +0X000000001"), SV("answer is {:+#012X}"), true));

  assert(check(SV("answer is 000000000000"), SV("answer is {:#012d}"), false));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012b}"), false));
  assert(check(SV("answer is 0b0000000000"), SV("answer is {:#012b}"), false));
  assert(check(SV("answer is 0B0000000000"), SV("answer is {:#012B}"), false));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012o}"), false));
  assert(check(SV("answer is 000000000000"), SV("answer is {:#012o}"), false));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012x}"), false));
  assert(check(SV("answer is 0x0000000000"), SV("answer is {:#012x}"), false));
  assert(check(SV("answer is 000000000000"), SV("answer is {:012X}"), false));
  assert(check(SV("answer is 0X0000000000"), SV("answer is {:#012X}"), false));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0}"), true));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42}"), true));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("bBcdosxX"))
  // {
  //   assert(check_exception("The type option contains an invalid value for a bool formatting argument", fmt, true));
  // }
}

TEST_FUNC void test()
{
  test_bool<char>();
  test_bool_as_integer<char>();
#if _CCCL_HAS_WCHAR_T()
  test_bool<wchar_t>();
  test_bool_as_integer<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
