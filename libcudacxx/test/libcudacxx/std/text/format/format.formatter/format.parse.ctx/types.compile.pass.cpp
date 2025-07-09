//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// Class typedefs:
// template<class charT>
// class basic_format_parse_context {
// public:
//   using char_type = charT;
//   using const_iterator = typename basic_string_view<charT>::const_iterator;
//   using iterator = const_iterator;
// }
//
// Namespace std typedefs:
// using format_parse_context = basic_format_parse_context<char>;
// using wformat_parse_context = basic_format_parse_context<wchar_t>;

#include <cuda/std/__format_>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class CharT>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::char_type, CharT>);
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::const_iterator,
                                     typename cuda::std::basic_string_view<CharT>::const_iterator>);
  static_assert(cuda::std::is_same_v<typename cuda::std::basic_format_parse_context<CharT>::iterator,
                                     typename cuda::std::basic_format_parse_context<CharT>::const_iterator>);
}

__host__ __device__ constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

static_assert(cuda::std::is_same_v<cuda::std::format_parse_context, cuda::std::basic_format_parse_context<char>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<cuda::std::wformat_parse_context, cuda::std::basic_format_parse_context<wchar_t>>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  static_assert(test());
  return 0;
}
