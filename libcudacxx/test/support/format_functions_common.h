//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H
#define TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H

// Contains the common part of the formatter tests for different papers.

#include <cuda/std/__format_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <cuda/std/cstring>
#include <cuda/std/ranges>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

#define SV(S)                         \
  cuda::std::basic_string_view<CharT> \
  {                                   \
    TEST_STRLIT(CharT, S)             \
  }
#define CSTR(S) TEST_STRLIT(CharT, S)

template <class T>
struct context
{};

template <>
struct context<char>
{
  using type = cuda::std::format_context;
};

#if _CCCL_HAS_WCHAR_T()
template <>
struct context<wchar_t>
{
  using type = cuda::std::wformat_context;
};
#endif // _CCCL_HAS_WCHAR_T()

template <class T>
using context_t = typename context<T>::type;

// A user-defined type used to test the handle formatter.
enum class status : cuda::std::uint16_t
{
  foo    = 0xAAAA,
  bar    = 0x5555,
  foobar = 0xAA55
};

// The formatter for a user-defined type used to test the handle formatter.
template <class CharT>
struct cuda::std::formatter<status, CharT>
{
  // During the 2023 Issaquah meeting LEWG made it clear a formatter is
  // required to call its parse function. LWG3892 Adds the wording for that
  // requirement. Therefore this formatter is initialized in an invalid state.
  // A call to parse sets it in a valid state and a call to format validates
  // the state.
  int type = -1;

  TEST_FUNC constexpr auto parse(basic_format_parse_context<CharT>& parse_ctx) -> decltype(parse_ctx.begin())
  {
    auto it = parse_ctx.begin();
    type    = 0;
    if (it == parse_ctx.end())
    {
      return it;
    }

    switch (*it)
    {
      case CharT('x'):
        break;
      case CharT('X'):
        type = 1;
        break;
      case CharT('s'):
        type = 2;
        break;
      case CharT('}'):
        return it;
      default:
        assert(false);
    }

    ++it;
    if (it != parse_ctx.end() && *it != CharT('}'))
    {
      assert(false);
    }

    return it;
  }

  template <class Out>
  TEST_FUNC auto format(status s, basic_format_context<Out, CharT>& ctx) const -> decltype(ctx.out())
  {
    const char* names[] = {"foo", "bar", "foobar"};
    char buffer[7];
    const char* result_begin = names[0];
    const char* result_end   = names[0];
    switch (type)
    {
      case -1:
        assert(false);

      case 0:
        result_begin = buffer;
        buffer[0]    = '0';
        buffer[1]    = 'x';
        result_end =
          cuda::std::to_chars(&buffer[2], cuda::std::end(buffer), static_cast<cuda::std::uint16_t>(s), 16).ptr;
        buffer[6] = '\0';
        break;

      case 1:
        result_begin = buffer;
        buffer[0]    = '0';
        buffer[1]    = 'X';
        result_end =
          cuda::std::to_chars(&buffer[2], cuda::std::end(buffer), static_cast<cuda::std::uint16_t>(s), 16).ptr;
        cuda::std::transform(static_cast<const char*>(&buffer[2]), result_end, &buffer[2], [](char c) {
          if (c >= 'a' && c <= 'z')
          {
            return static_cast<char>(c - ('a' - 'A'));
          }
          return c;
        });
        buffer[6] = '\0';
        break;

      case 2:
        switch (s)
        {
          case status::foo:
            result_begin = names[0];
            break;
          case status::bar:
            result_begin = names[1];
            break;
          case status::foobar:
            result_begin = names[2];
            break;
        }
        result_end = result_begin + cuda::std::strlen(result_begin);
        break;
    }

    return cuda::std::copy(result_begin, result_end, ctx.out());
  }
};

struct parse_call_validator
{
  struct parse_function_not_called
  {};

  TEST_FUNC friend constexpr auto operator==(const parse_call_validator& lhs, const parse_call_validator& rhs)
  {
    return &lhs == &rhs;
  }
};

// The formatter for a user-defined type used to test the handle formatter.
//
// Like cuda::std::formatter<status, CharT> this formatter validates that parse is
// called. This formatter is intended to be used when the formatter's parse is
// called directly and not with format. In that case the format-spec does not
// require a terminating }. The tests must be written in a fashion where this
// formatter is always called with an empty format-spec. This requirement
// allows testing of certain code paths that are never reached by using a
// well-formed format-string in the format functions.
template <class CharT>
struct cuda::std::formatter<parse_call_validator, CharT>
{
  bool parse_called{false};

  TEST_FUNC constexpr auto parse(basic_format_parse_context<CharT>& parse_ctx) -> decltype(parse_ctx.begin())
  {
    assert(parse_ctx.begin() == parse_ctx.end());
    parse_called = true;
    return parse_ctx.begin();
  }

  template <class Ctx>
  TEST_FUNC auto format(parse_call_validator, Ctx& ctx) const -> decltype(ctx.out())
  {
    if (!parse_called)
    {
      assert(false);
    }
    return ctx.out();
  }
};

// Creates format string for the invalid types.
//
// valid contains a list of types that are valid.
// - The type ?s is the only type requiring 2 characters, use S for that type.
// - Whether n is a type or not depends on the context, is is always used.
//
// The return value is a collection of basic_strings, instead of
// basic_string_views since the values are temporaries.
namespace detail
{
template <class CharT, cuda::std::size_t N>
TEST_FUNC auto get_colons()
{
  return cuda::std::inplace_vector<CharT, N>(N, CharT(':'));
}

TEST_FUNC constexpr cuda::std::string_view get_format_types()
{
  return "aAbBcdeEfFgGopPsxX"
#if TEST_STD_VER > 20
         "?"
#endif
    ;
}

// template <class CharT, /*format_types types,*/ size_t N>
// cuda::std::vector<cuda::std::basic_string<CharT>> fmt_invalid_types(cuda::std::string_view valid) {
//   // cuda::std::ranges::to is not available in C++20.
//   cuda::std::vector<cuda::std::basic_string<CharT>> result;
//   cuda::std::ranges::copy(
//       get_format_types() | cuda::std::views::filter([&](char type) { return valid.find(type) ==
//       cuda::std::string_view::npos; }) |
//           cuda::std::views::transform([&](char type) { return cuda::std::format(SV("{{{}{}}}"), get_colons<CharT,
//           N>(), type); }),
//       cuda::std::back_inserter(result));
//   return result;
// }
} // namespace detail

// Creates format string for the invalid types.
//
// valid contains a list of types that are valid.
//
// The return value is a collection of basic_strings, instead of
// basic_string_views since the values are temporaries.
// template <class CharT>
// cuda::std::vector<cuda::std::basic_string<CharT>> fmt_invalid_types(cuda::std::string_view valid) {
//   return detail::fmt_invalid_types<CharT, 1>(valid);
// }

// Like fmt_invalid_types but when the format spec is for an underlying formatter.
// template <class CharT>
// cuda::std::vector<cuda::std::basic_string<CharT>> fmt_invalid_nested_types(cuda::std::string_view valid) {
//   return detail::fmt_invalid_types<CharT, 2>(valid);
// }

#if _CCCL_HAS_WCHAR_T()
template <class CharT, class... Args>
using test_format_string =
  cuda::std::conditional_t<cuda::std::is_same_v<CharT, char>,
                           cuda::std::format_string<Args...>,
                           cuda::std::wformat_string<Args...>>;
#else
template <class CharT, class... Args>
using test_format_string = cuda::std::format_string<Args...>;
#endif

#endif // TEST_SUPPORT_FORMAT_FUNCTIONS_COMMON_H
