//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
template <class CharT, class... Args>
TEST_FUNC bool
check(cuda::std::basic_string_view<CharT> expected, cuda::std::basic_string_view<CharT> fmt, Args&&... args);
template <class CharT, class... Args>
TEST_FUNC bool check_exception(cuda::std::string_view what, cuda::std::basic_string_view<CharT> fmt, Args&&... args);

template <class CharT, class T>
TEST_FUNC void test_pointer()
{
  // *** align-fill & width ***
  assert(check(SV("answer is '   0x0'"), SV("answer is '{:6}'"), T(nullptr)));
  assert(check(SV("answer is '   0x0'"), SV("answer is '{:>6}'"), T(nullptr)));
  assert(check(SV("answer is '0x0   '"), SV("answer is '{:<6}'"), T(nullptr)));
  assert(check(SV("answer is ' 0x0  '"), SV("answer is '{:^6}'"), T(nullptr)));

  // The fill character ':' is allowed here (P0645) but not in ranges (P2286).
  assert(check(SV("answer is ':::0x0'"), SV("answer is '{::>6}'"), T(nullptr)));
  assert(check(SV("answer is '0x0:::'"), SV("answer is '{::<6}'"), T(nullptr)));
  assert(check(SV("answer is ':0x0::'"), SV("answer is '{::^6}'"), T(nullptr)));

  // Test whether zero padding is ignored
  assert(check(SV("answer is ':::0x0'"), SV("answer is '{::>06}'"), T(nullptr)));
  assert(check(SV("answer is '0x0:::'"), SV("answer is '{::<06}'"), T(nullptr)));
  assert(check(SV("answer is ':0x0::'"), SV("answer is '{::^06}'"), T(nullptr)));

  // *** Sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), T(nullptr)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), T(nullptr)));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), T(nullptr)));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), T(nullptr)));

  // *** zero-padding ***
  assert(check(SV("answer is '0x0000'"), SV("answer is '{:06}'"), T(nullptr)));
  assert(check(SV("answer is '0x0000'"), SV("answer is '{:06p}'"), T(nullptr)));
  assert(check(SV("answer is '0X0000'"), SV("answer is '{:06P}'"), T(nullptr)));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), nullptr));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.0}"), nullptr));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.42}"), nullptr));

  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), nullptr));
  assert(
    check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), nullptr, true));
  assert(
    check_exception("The format specifier should consume the input or end with a '}'", SV("{:.{}}"), nullptr, 1.0));

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("pP"))
  // {
  //   check_exception("The type option contains an invalid value for a pointer formatting argument", fmt, T(nullptr));
  // }
}

template <class CharT>
TEST_FUNC void test()
{
  test_pointer<CharT, cuda::std::nullptr_t>();
  test_pointer<CharT, void*>();
  test_pointer<CharT, const void*>();
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
