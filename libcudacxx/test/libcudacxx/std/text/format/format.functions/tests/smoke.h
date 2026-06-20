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
TEST_FUNC void smoke_test()
{
  // *** Test escaping  ***
  assert(check(SV("{"), SV("{{")));
  assert(check(SV("}"), SV("}}")));
  assert(check(SV("{:^}"), SV("{{:^}}")));
  assert(check(SV("{: ^}"), SV("{{:{}^}}"), CharT(' ')));
  assert(check(SV("{:{}^}"), SV("{{:{{}}^}}")));
  assert(check(SV("{:{ }^}"), SV("{{:{{{}}}^}}"), CharT(' ')));

  // *** Test argument ID ***
  assert(check(SV("hello false true"), SV("hello {0:} {1:}"), false, true));
  assert(check(SV("hello true false"), SV("hello {1:} {0:}"), false, true));

  // *** Test many arguments ***

  // [format.args]/1
  // An instance of basic_format_args provides access to formatting arguments.
  // Implementations should optimize the representation of basic_format_args
  // for a small number of formatting arguments.
  //
  // These's no guidances what "a small number of formatting arguments" is.
  // - fmtlib uses a 15 elements
  // - libc++ uses 12 elements
  // - MSVC STL uses a different approach regardless of the number of arguments
  // - libstdc++ has no implementation yet
  // fmtlib and libc++ use a similar approach, this approach can support 16
  // elements (based on design choices both support less elements). This test
  // makes sure "the large number of formatting arguments" code path is tested.
  assert(check(
    SV("1234567890\t1234567890"),
    SV("{}{}{}{}{}{}{}{}{}{}\t{}{}{}{}{}{}{}{}{}{}"),
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    0));

  // *** Test invalid format strings ***
  assert(check_exception("The format string terminates at a '{'", SV("{")));
  assert(check_exception("The argument index value is too large for the number of arguments supplied", SV("{:")));
  assert(check_exception("The replacement field misses a terminating '}'", SV("{:"), 42));

  assert(check_exception("The argument index should end with a ':' or a '}'", SV("{0")));
  assert(check_exception("The argument index value is too large for the number of arguments supplied", SV("{0:")));
  assert(check_exception("The replacement field misses a terminating '}'", SV("{0:"), 42));

  assert(check_exception("The format string contains an invalid escape sequence", SV("}")));
  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}-}"), 42));

  assert(check_exception("The format string contains an invalid escape sequence", SV("} ")));
  assert(check_exception("The argument index starts with an invalid character", SV("{-"), 42));
  assert(check_exception("The argument index value is too large for the number of arguments supplied", SV("hello {}")));
  assert(
    check_exception("The argument index value is too large for the number of arguments supplied", SV("hello {0}")));
  assert(
    check_exception("The argument index value is too large for the number of arguments supplied", SV("hello {1}"), 42));

  // *** Test char format argument ***
  // The `char` to `wchar_t` formatting is tested separately.
  assert(check(
    SV("hello 09azAZ!"),
    SV("hello {}{}{}{}{}{}{}"),
    CharT('0'),
    CharT('9'),
    CharT('a'),
    CharT('z'),
    CharT('A'),
    CharT('Z'),
    CharT('!')));

  // *** Test string format argument ***
  {
    CharT buffer[] = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    CharT* data    = buffer;
    assert(check(SV("hello 09azAZ!"), SV("hello {}"), data));
  }
  {
    CharT buffer[]    = {CharT('0'), CharT('9'), CharT('a'), CharT('z'), CharT('A'), CharT('Z'), CharT('!'), 0};
    const CharT* data = buffer;
    assert(check(SV("hello 09azAZ!"), SV("hello {}"), data));
  }
  {
    // https://llvm.org/PR115935
    // Contents after the embedded null character are discarded.
    CharT buffer[] = {CharT('a'), CharT('b'), CharT('c'), 0, CharT('d'), CharT('e'), CharT('f'), 0};
    assert(check(SV("hello abc"), SV("hello {}"), buffer));
    // Even when the last element of the array is not null character.
    CharT buffer2[] = {CharT('a'), CharT('b'), CharT('c'), 0, CharT('d'), CharT('e'), CharT('f')};
    assert(check(SV("hello abc"), SV("hello {}"), buffer2));
  }
  {
    // todo(dabayer): Host stdlib interop.
    // std::basic_string<CharT> data = TEST_STRLIT(CharT, "world");
    // assert(check(SV("hello world"), SV("hello {}"), data));
  }
  {
    // todo(dabayer): Host stdlib interop.
    // std::basic_string_view<CharT> data = TEST_STRLIT(CharT, "world");
    // assert(check(SV("hello world"), SV("hello {}"), data));
  }
  {
    cuda::std::basic_string_view<CharT> data = TEST_STRLIT(CharT, "world");
    assert(check(SV("hello world"), SV("hello {}"), data));
  }

  // *** Test Boolean format argument ***
  assert(check(SV("hello false true"), SV("hello {} {}"), false, true));

  // *** Test signed integral format argument ***
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<signed char>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<short>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<int>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<long>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<long long>(42)));
#if _CCCL_HAS_INT128()
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<__int128_t>(42)));
#endif // _CCCL_HAS_INT128()

  // ** Test unsigned integral format argument ***
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<unsigned char>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<unsigned short>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<unsigned>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<unsigned long>(42)));
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<unsigned long long>(42)));
#if _CCCL_HAS_INT128()
  assert(check(SV("hello 42"), SV("hello {}"), static_cast<__uint128_t>(42)));
#endif // _CCCL_HAS_INT128()

  // todo(dabayer): Make formatters for fp types work.
  // *** Test floating point format argument ***
  //   assert(check(SV("hello 42"), SV("hello {}"), static_cast<float>(42)));
  //   assert(check(SV("hello 42"), SV("hello {}"), static_cast<double>(42)));
  // #if _CCCL_HAS_LONG_DOUBLE()
  //   assert(check(SV("hello 42"), SV("hello {}"), static_cast<long double>(42)));
  // #endif // _CCCL_HAS_LONG_DOUBLE()

  // *** Test pointer formatter argument ***
  assert(check(SV("hello 0x0"), SV("hello {}"), nullptr));
  assert(check(SV("hello 0x42"), SV("hello {}"), reinterpret_cast<void*>(0x42)));
  assert(check(SV("hello 0x42"), SV("hello {}"), reinterpret_cast<const void*>(0x42)));

  // *** Test handle formatter argument ***
  assert(check(SV("answer is '0xaaaa'"), SV("answer is '{}'"), status::foo));
  assert(check(SV("answer is '0x5555'"), SV("answer is '{:x}'"), status::bar));
  assert(check(SV("answer is 'foobar'"), SV("answer is '{:s}'"), status::foobar));
}

TEST_FUNC void test()
{
  smoke_test<char>();
#if _CCCL_HAS_WCHAR_T()
  smoke_test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
