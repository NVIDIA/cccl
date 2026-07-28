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
TEST_FUNC void test_handle()
{
  // *** Valid permutations ***
  assert(check(SV("answer is '0xaaaa'"), SV("answer is '{}'"), status::foo));
  assert(check(SV("answer is '0xaaaa'"), SV("answer is '{:x}'"), status::foo));
  assert(check(SV("answer is '0XAAAA'"), SV("answer is '{:X}'"), status::foo));
  assert(check(SV("answer is 'foo'"), SV("answer is '{:s}'"), status::foo));

  assert(check(SV("answer is '0x5555'"), SV("answer is '{}'"), status::bar));
  assert(check(SV("answer is '0x5555'"), SV("answer is '{:x}'"), status::bar));
  assert(check(SV("answer is '0X5555'"), SV("answer is '{:X}'"), status::bar));
  assert(check(SV("answer is 'bar'"), SV("answer is '{:s}'"), status::bar));

  assert(check(SV("answer is '0xaa55'"), SV("answer is '{}'"), status::foobar));
  assert(check(SV("answer is '0xaa55'"), SV("answer is '{:x}'"), status::foobar));
  assert(check(SV("answer is '0XAA55'"), SV("answer is '{:X}'"), status::foobar));
  assert(check(SV("answer is 'foobar'"), SV("answer is '{:s}'"), status::foobar));

  // P2418 Changed the argument from a const reference to a forwarding reference.
  // This mainly affects handle classes, however since we use an abstraction
  // layer here it's "tricky" to verify whether this test would do the "right"
  // thing. So these tests are done separately.

  // todo(dabayer): Test invalid types.
  // *** type ***
  // for (const auto& fmt : invalid_types<CharT>("xXs"))
  // {
  //   check_exception("The type option contains an invalid value for a status formatting argument", fmt, status::foo);
  // }
}

TEST_FUNC void test()
{
  test_handle<char>();
#if _CCCL_HAS_WCHAR_T()
  test_handle<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
