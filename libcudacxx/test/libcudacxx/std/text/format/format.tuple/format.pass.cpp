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

// template<class FormatContext>
//   typename FormatContext::iterator
//     format(see below& elems, FormatContext& ctx) const;

// Note this tests the basics of this function. It's tested in more detail in
// the format functions tests.

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "format_functions_common.h"
#include "literal.h"
#include "test_macros.h"

template <class StringViewT, class Arg>
TEST_FUNC void test(StringViewT expected, Arg arg)
{
  using CharT      = typename StringViewT::value_type;
  using Container  = cuda::std::inplace_vector<CharT, 256>;
  using OutIt      = cuda::std::__back_insert_iterator<Container>;
  using FormatCtxT = cuda::std::basic_format_context<OutIt, CharT>;

  const cuda::std::formatter<Arg, CharT> formatter;

  Container result;
  OutIt out = cuda::std::__back_insert_iterator{result};
  FormatCtxT format_ctx =
    cuda::std::__fmt_make_format_context<OutIt, CharT>(out, cuda::std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(StringViewT{result} == expected);
}

template <class CharT>
TEST_FUNC void test_assure_parse_is_called(cuda::std::basic_string_view<CharT> fmt)
{
  using Container  = cuda::std::inplace_vector<CharT, 256>;
  using OutIt      = cuda::std::__back_insert_iterator<Container>;
  using FormatCtxT = cuda::std::basic_format_context<OutIt, CharT>;
  cuda::std::pair<parse_call_validator, parse_call_validator> arg;

  Container result;
  OutIt out = cuda::std::__back_insert_iterator{result};
  FormatCtxT format_ctx =
    cuda::std::__fmt_make_format_context<OutIt, CharT>(out, cuda::std::make_format_args<FormatCtxT>(arg));

  cuda::std::formatter<decltype(arg), CharT> formatter;
  cuda::std::basic_format_parse_context<CharT> ctx{fmt};

  formatter.parse(ctx);
  formatter.format(arg, format_ctx);
}

template <class CharT>
TEST_FUNC void test_assure_parse_is_called()
{
  using Container  = cuda::std::inplace_vector<CharT, 256>;
  using OutIt      = cuda::std::__back_insert_iterator<Container>;
  using FormatCtxT = cuda::std::basic_format_context<OutIt, CharT>;
  cuda::std::pair<parse_call_validator, parse_call_validator> arg;

  Container result;
  OutIt out = cuda::std::__back_insert_iterator{result};
  [[maybe_unused]] FormatCtxT format_ctx =
    cuda::std::__fmt_make_format_context<OutIt, CharT>(out, cuda::std::make_format_args<FormatCtxT>(arg));

#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST,
               ({ // parse not called
                 [[maybe_unused]] const cuda::std::formatter<decltype(arg), CharT> formatter;
                 try
                 {
                   formatter.format(arg, format_ctx);
                   assert(false);
                 }
                 catch (const parse_call_validator::parse_function_not_called&)
                 {
                   assert(true);
                 }
                 catch (...)
                 {
                   assert(false);
                 }
               }))
#endif // _CCCL_HAS_EXCEPTIONS()

  test_assure_parse_is_called(cuda::std::basic_string_view{TEST_STRLIT(CharT, "5")});
  test_assure_parse_is_called(cuda::std::basic_string_view{TEST_STRLIT(CharT, "n")});
  test_assure_parse_is_called(cuda::std::basic_string_view{TEST_STRLIT(CharT, "m")});
  test_assure_parse_is_called(cuda::std::basic_string_view{TEST_STRLIT(CharT, "5n")});
  test_assure_parse_is_called(cuda::std::basic_string_view{TEST_STRLIT(CharT, "5m")});
}

template <class CharT>
TEST_FUNC void test()
{
  test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1)")}, cuda::std::tuple<int>{1});
  test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1, 1)")}, cuda::std::tuple<int, CharT>{1, CharT{'1'}});
  test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1, 1)")}, cuda::std::pair<int, CharT>{1, CharT{'1'}});
  test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1, 1, true)")},
       cuda::std::tuple<int, CharT, bool>{1, CharT{'1'}, true});

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1, 1)")},
                      std::tuple<int, CharT>{1, CharT{'1'}});
                 test(cuda::std::basic_string_view{TEST_STRLIT(CharT, "(1, 1)")}, std::pair<int, CharT>{1, CharT{'1'}});
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  test_assure_parse_is_called<CharT>();
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
