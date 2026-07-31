//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// class range_formatter
// template<class charT, formattable<charT>... Ts>
//   struct formatter<pair-or-tuple<Ts...>, charT>

// constexpr void set_separator(basic_string_view<charT> sep) noexcept;

// Note this tests the basics of this function. It's tested in more detail in
// the format functions tests.

#include <cuda/std/__format_>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "literal.h"
#include "test_macros.h"

template <class CharT, class Arg>
TEST_FUNC constexpr void test()
{
  cuda::std::formatter<Arg, CharT> formatter;

  static_assert(cuda::std::is_same_v<void, decltype(formatter.set_separator(cuda::std::basic_string_view<CharT>{}))>);
  static_assert(noexcept(formatter.set_separator(cuda::std::basic_string_view<CharT>{})));

  formatter.set_separator(cuda::std::basic_string_view{TEST_STRLIT(CharT, "sep")});

  // Note there is no direct way to validate this function modified the object.
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
