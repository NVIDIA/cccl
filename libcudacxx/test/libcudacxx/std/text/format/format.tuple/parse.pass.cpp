//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT, formattable<charT>... Ts>
//   struct formatter<pair-or-tuple<Ts...>, charT>

// template<class ParseContext>
//   constexpr typename ParseContext::iterator
//     parse(ParseContext& ctx);

// Note this tests the basics of this function. It's tested in more detail in
// the format functions tests.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/memory>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "literal.h"
#include "test_macros.h"

template <class Arg, class StringViewT>
TEST_FUNC constexpr void test(StringViewT fmt, cuda::std::size_t offset)
{
  using CharT = typename StringViewT::value_type;

  cuda::std::basic_format_parse_context parse_ctx{fmt};
  cuda::std::formatter<Arg, CharT> formatter;

  static_assert(cuda::std::semiregular<decltype(formatter)>);

  static_assert(cuda::std::is_same_v<typename StringViewT::iterator, decltype(formatter.parse(parse_ctx))>);
  static_assert(!noexcept(formatter.parse(parse_ctx)));

  auto it = formatter.parse(parse_ctx);
  // cuda::std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(cuda::std::to_address(it) == cuda::std::to_address(fmt.end()) - offset);
}

template <class CharT, class Arg>
TEST_FUNC constexpr void test()
{
  test<Arg>(cuda::std::basic_string_view{TEST_STRLIT(CharT, "")}, 0);
  test<Arg>(cuda::std::basic_string_view{TEST_STRLIT(CharT, "42")}, 0);

  test<Arg>(cuda::std::basic_string_view{TEST_STRLIT(CharT, "}")}, 1);
  test<Arg>(cuda::std::basic_string_view{TEST_STRLIT(CharT, "42}")}, 1);
}

template <class CharT>
TEST_FUNC constexpr void test()
{
  test<CharT, cuda::std::tuple<int>>();
  test<CharT, cuda::std::tuple<int, CharT>>();
  test<CharT, cuda::std::pair<int, CharT>>();
  test<CharT, cuda::std::tuple<int, CharT, bool>>();

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 test<CharT, std::tuple<int, CharT>>();
                 test<CharT, std::pair<int, CharT>>();
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()
}

TEST_FUNC constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
