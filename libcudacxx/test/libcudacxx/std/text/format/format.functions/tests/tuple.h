//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__format_>
#include <cuda/std/concepts>
#include <cuda/std/tuple>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
TEST_FUNC bool check(...);
TEST_FUNC bool check_exception(...);

enum class color
{
  black,
  red,
  gold
};

template <class CharT>
struct cuda::std::formatter<color, CharT> : cuda::std::formatter<basic_string_view<CharT>, CharT>
{
  template <class Ctx>
  TEST_FUNC auto format(color c, Ctx& ctx) const
  {
    constexpr basic_string_view<CharT> color_names[]{
      TEST_STRLIT(CharT, "black"), TEST_STRLIT(CharT, "red"), TEST_STRLIT(CharT, "gold")};
    return formatter<basic_string_view<CharT>, CharT>::format(color_names[static_cast<int>(c)], ctx);
  }
};

//
// Generic tests for a tuple and pair with two elements.
//
template <class CharT, class TupleOrPair>
TEST_FUNC void test_tuple_or_pair_int_int(TupleOrPair&& input)
{
  assert(check(SV("(42, 99)"), SV("{}"), input));
  assert(check(SV("(42, 99)^42"), SV("{}^42"), input));
  assert(check(SV("(42, 99)^42"), SV("{:}^42"), input));

  // *** align-fill & width ***
  assert(check(SV("(42, 99)     "), SV("{:13}"), input));
  assert(check(SV("(42, 99)*****"), SV("{:*<13}"), input));
  assert(check(SV("__(42, 99)___"), SV("{:_^13}"), input));
  assert(check(SV("#####(42, 99)"), SV("{:#>13}"), input));

  assert(check(SV("(42, 99)     "), SV("{:{}}"), input, 13));
  assert(check(SV("(42, 99)*****"), SV("{:*<{}}"), input, 13));
  assert(check(SV("__(42, 99)___"), SV("{:_^{}}"), input, 13));
  assert(check(SV("#####(42, 99)"), SV("{:#>{}}"), input, 13));

  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input));
  assert(check_exception("The fill option contains an invalid value", SV("{:{<}"), input));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), input));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), input));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("{:0}"), input));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), input));

  // *** type ***
  assert(check(SV("__42: 99___"), SV("{:_^11m}"), input));
  assert(check(SV("__42, 99___"), SV("{:_^11n}"), input));

  // todo(dabayer): Test invalid specifiers.
  // for (CharT c : SV("aAbBcdeEfFgGopPsxX?")) {
  //   assert(check_exception("The format specifier should consume the input or end with a '}'",
  //                   cuda::std::basic_string_view{STR("{:") + c + STR("}")},
  //                   input));
  // }
}

template <class CharT, class TupleOrPair>
TEST_FUNC void test_tuple_or_pair_int_string(TupleOrPair&& input)
{
  assert(check(SV("(42, \"hello\")"), SV("{}"), input));
  assert(check(SV("(42, \"hello\")^42"), SV("{}^42"), input));
  assert(check(SV("(42, \"hello\")^42"), SV("{:}^42"), input));

  // *** align-fill & width ***
  assert(check(SV("(42, \"hello\")     "), SV("{:18}"), input));
  assert(check(SV("(42, \"hello\")*****"), SV("{:*<18}"), input));
  assert(check(SV("__(42, \"hello\")___"), SV("{:_^18}"), input));
  assert(check(SV("#####(42, \"hello\")"), SV("{:#>18}"), input));

  assert(check(SV("(42, \"hello\")     "), SV("{:{}}"), input, 18));
  assert(check(SV("(42, \"hello\")*****"), SV("{:*<{}}"), input, 18));
  assert(check(SV("__(42, \"hello\")___"), SV("{:_^{}}"), input, 18));
  assert(check(SV("#####(42, \"hello\")"), SV("{:#>{}}"), input, 18));

  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input));
  assert(check_exception("The fill option contains an invalid value", SV("{:{<}"), input));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), input));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), input));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("{:0}"), input));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), input));

  // *** type ***
  assert(check(SV("__42: \"hello\"___"), SV("{:_^16m}"), input));
  assert(check(SV("__42, \"hello\"___"), SV("{:_^16n}"), input));

  // todo(dabayer): Test invalid specifiers.
  // for (CharT c : SV("aAbBcdeEfFgGopPsxX?")) {
  //   assert(check_exception("The format specifier should consume the input or end with a '}'",
  //                   cuda::std::basic_string_view{STR("{:") + c + STR("}")},
  //                   input));
  // }
}

template <class CharT, class TupleOrPair>
TEST_FUNC void test_escaping(TupleOrPair&& input)
{
  static_assert(cuda::std::same_as<cuda::std::remove_cvref_t<decltype(cuda::std::get<0>(input))>, CharT>);
  static_assert(cuda::std::same_as<cuda::std::remove_cvref_t<decltype(cuda::std::get<1>(input))>,
                                   cuda::std::basic_string_view<CharT>>);

  assert(check(SV(R"(('*', ""))"), SV("{}"), input));

  // Char
  cuda::std::get<0>(input) = CharT('\t');
  assert(check(SV(R"(('\t', ""))"), SV("{}"), input));
  cuda::std::get<0>(input) = CharT('\n');
  assert(check(SV(R"(('\n', ""))"), SV("{}"), input));
  cuda::std::get<0>(input) = CharT('\0');
  assert(check(SV(R"(('\u{0}', ""))"), SV("{}"), input));

  // String
  cuda::std::get<0>(input) = CharT('*');
  cuda::std::get<1>(input) = SV("hellö");
  assert(check(SV("('*', \"hellö\")"), SV("{}"), input));
}

//
// pair tests
//

template <class CharT>
TEST_FUNC void test_pair_int_int()
{
  test_tuple_or_pair_int_int<CharT>(cuda::std::make_pair(42, 99));
}

template <class CharT>
TEST_FUNC void test_pair_int_string()
{
  test_tuple_or_pair_int_string<CharT>(cuda::std::make_pair(42, SV("hello")));
  test_tuple_or_pair_int_string<CharT>(cuda::std::make_pair(42, CSTR("hello")));
}

//
// tuple tests
//

template <class CharT>
TEST_FUNC void test_tuple_int()
{
  auto input = cuda::std::make_tuple(42);

  assert(check(SV("(42)"), SV("{}"), input));
  assert(check(SV("(42)^42"), SV("{}^42"), input));
  assert(check(SV("(42)^42"), SV("{:}^42"), input));

  // *** align-fill & width ***
  assert(check(SV("(42)     "), SV("{:9}"), input));
  assert(check(SV("(42)*****"), SV("{:*<9}"), input));
  assert(check(SV("__(42)___"), SV("{:_^9}"), input));
  assert(check(SV("#####(42)"), SV("{:#>9}"), input));

  assert(check(SV("(42)     "), SV("{:{}}"), input, 9));
  assert(check(SV("(42)*****"), SV("{:*<{}}"), input, 9));
  assert(check(SV("__(42)___"), SV("{:_^{}}"), input, 9));
  assert(check(SV("#####(42)"), SV("{:#>{}}"), input, 9));

  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input));
  assert(check_exception("The fill option contains an invalid value", SV("{:{<}"), input));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), input));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), input));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("{:0}"), input));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), input));

  // *** type ***
  assert(check_exception("Type m requires a pair or a tuple with two elements", SV("{:m}"), input));
  assert(check(SV("__42___"), SV("{:_^7n}"), input));

  // todo(dabayer): Test invalid specifiers.
  // for (CharT c : SV("aAbBcdeEfFgGopPsxX?")) {
  //   check_exception("The format specifier should consume the input or end with a '}'",
  //                   cuda::std::basic_string_view{STR("{:") + c + STR("}")},
  //                   input);
  // }
}

template <class CharT>
TEST_FUNC void test_tuple_int_string_color()
{
  const auto input = cuda::std::make_tuple(42, SV("hello"), color::red);

  assert(check(SV("(42, \"hello\", \"red\")"), SV("{}"), input));
  assert(check(SV("(42, \"hello\", \"red\")^42"), SV("{}^42"), input));
  assert(check(SV("(42, \"hello\", \"red\")^42"), SV("{:}^42"), input));

  // *** align-fill & width ***
  assert(check(SV("(42, \"hello\", \"red\")     "), SV("{:25}"), input));
  assert(check(SV("(42, \"hello\", \"red\")*****"), SV("{:*<25}"), input));
  assert(check(SV("__(42, \"hello\", \"red\")___"), SV("{:_^25}"), input));
  assert(check(SV("#####(42, \"hello\", \"red\")"), SV("{:#>25}"), input));

  assert(check(SV("(42, \"hello\", \"red\")     "), SV("{:{}}"), input, 25));
  assert(check(SV("(42, \"hello\", \"red\")*****"), SV("{:*<{}}"), input, 25));
  assert(check(SV("__(42, \"hello\", \"red\")___"), SV("{:_^{}}"), input, 25));
  assert(check(SV("#####(42, \"hello\", \"red\")"), SV("{:#>{}}"), input, 25));

  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input));
  assert(check_exception("The fill option contains an invalid value", SV("{:{<}"), input));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), input));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), input));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("{:0}"), input));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), input));

  // *** type ***
  assert(check_exception("Type m requires a pair or a tuple with two elements", SV("{:m}"), input));
  assert(check(SV("__42, \"hello\", \"red\"___"), SV("{:_^23n}"), input));

  // todo(dabayer): Test invalid specifiers.
  // for (CharT c : SV("aAbBcdeEfFgGopPsxX?")) {
  //   assert(check_exception("The format specifier should consume the input or end with a '}'",
  //                   cuda::std::basic_string_view{STR("{:") + c + STR("}")},
  //                   input));
  // }
}

template <class CharT>
TEST_FUNC void test_tuple_int_int()
{
  test_tuple_or_pair_int_int<CharT>(cuda::std::make_tuple(42, 99));
}

template <class CharT>
TEST_FUNC void test_tuple_int_string()
{
  test_tuple_or_pair_int_string<CharT>(cuda::std::make_tuple(42, SV("hello")));
  test_tuple_or_pair_int_string<CharT>(cuda::std::make_tuple(42, CSTR("hello")));
}

//
// nested tests
//

template <class CharT, class Nested>
TEST_FUNC void test_nested(Nested&& input)
{
  // [format.formatter.spec]/2
  //   A debug-enabled specialization of formatter additionally provides a
  //   public, constexpr, non-static member function set_debug_format()
  //   which modifies the state of the formatter to be as if the type of the
  //   std-format-spec parsed by the last call to parse were ?.
  // pair and tuple are not debug-enabled specializations to the
  // set_debug_format is not propagated. The paper
  //   P2733 Fix handling of empty specifiers in cuda::std::format
  // addressed this.

  assert(check(SV("(42, (\"hello\", \"red\"))"), SV("{}"), input));
  assert(check(SV("(42, (\"hello\", \"red\"))^42"), SV("{}^42"), input));
  assert(check(SV("(42, (\"hello\", \"red\"))^42"), SV("{:}^42"), input));

  // *** align-fill & width ***
  assert(check(SV("(42, (\"hello\", \"red\"))     "), SV("{:27}"), input));
  assert(check(SV("(42, (\"hello\", \"red\"))*****"), SV("{:*<27}"), input));
  assert(check(SV("__(42, (\"hello\", \"red\"))___"), SV("{:_^27}"), input));
  assert(check(SV("#####(42, (\"hello\", \"red\"))"), SV("{:#>27}"), input));

  assert(check(SV("(42, (\"hello\", \"red\"))     "), SV("{:{}}"), input, 27));
  assert(check(SV("(42, (\"hello\", \"red\"))*****"), SV("{:*<{}}"), input, 27));
  assert(check(SV("__(42, (\"hello\", \"red\"))___"), SV("{:_^{}}"), input, 27));
  assert(check(SV("#####(42, (\"hello\", \"red\"))"), SV("{:#>{}}"), input, 27));

  assert(check_exception("The format string contains an invalid escape sequence", SV("{:}<}"), input));
  assert(check_exception("The fill option contains an invalid value", SV("{:{<}"), input));

  // *** sign ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:-}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:+}"), input));
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{: }"), input));

  // *** alternate form ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:#}"), input));

  // *** zero-padding ***
  assert(check_exception("The width option should not have a leading zero", SV("{:0}"), input));

  // *** precision ***
  assert(check_exception("The format specifier should consume the input or end with a '}'", SV("{:.}"), input));

  // *** type ***
  assert(check(SV("__42: (\"hello\", \"red\")___"), SV("{:_^25m}"), input));
  assert(check(SV("__42, (\"hello\", \"red\")___"), SV("{:_^25n}"), input));

  // todo(dabayer): Test invalid specifiers.
  // for (CharT c : SV("aAbBcdeEfFgGopPsxX?")) {
  //   assert(check_exception("The format specifier should consume the input or end with a '}'",
  //                   cuda::std::basic_string_view{STR("{:") + c + STR("}")},
  //                   input));
  // }
}

template <class CharT>
TEST_FUNC void test()
{
  // todo(dabayer): Enable tuple with strings tests once debug formatting is implemented.

  test_pair_int_int<CharT>();
  // test_pair_int_string<CharT>();

  test_tuple_int<CharT>();
  test_tuple_int_int<CharT>();
  // test_tuple_int_string<CharT>();
  // test_tuple_int_string_color<CharT>();

  // test_nested<CharT>(cuda::std::make_pair(42, cuda::std::make_pair(SV("hello"), color::red)));
  // test_nested<CharT>(cuda::std::make_pair(42, cuda::std::make_tuple(SV("hello"), color::red)));
  // test_nested<CharT>(cuda::std::make_tuple(42, cuda::std::make_pair(SV("hello"), color::red)));
  // test_nested<CharT>(cuda::std::make_tuple(42, cuda::std::make_tuple(SV("hello"), color::red)));

  // test_escaping<CharT>(cuda::std::make_pair(CharT('*'), SV("")));
  // test_escaping<CharT>(cuda::std::make_tuple(CharT('*'), SV("")));

  // Test const ref-qualified types.
  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<int>{42}));
  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<const int>{42}));

  int answer = 42;
  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<int&>{answer}));
  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<const int&>{answer}));

  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<int&&>{42}));
  assert(check(SV("(42)"), SV("{}"), cuda::std::tuple<const int&&>{42}));

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 assert(check(SV("(42, true)"), SV("{}"), std::tuple{42, true}));
                 assert(check(SV("(42, true)"), SV("{}"), std::pair{42, true}));
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()
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
