//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT> struct dynamic-format-string {  // exposition-only
// private:
//   basic_string_view<charT> str;  // exposition-only
//
// public:
//   dynamic-format-string(basic_string_view<charT> s) noexcept : str(s) {}
//
//   dynamic-format-string(const dynamic-format-string&) = delete;
//   dynamic-format-string& operator=(const dynamic-format-string&) = delete;
// };
//
// dynamic-format-string<char> dynamic_format(string_view fmt) noexcept;
// dynamic-format-string<wchar_t> dynamic_format(wstring_view fmt) noexcept;
//
// Additional testing is done in format_to checkers

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class CharT>
TEST_FUNC void test_properties()
{
  static_assert(cuda::std::is_nothrow_convertible_v<cuda::std::basic_string_view<CharT>, T>);
  static_assert(cuda::std::is_nothrow_constructible_v<T, cuda::std::basic_string_view<CharT>>);

  static_assert(!cuda::std::copy_constructible<T>);
  static_assert(!cuda::std::is_copy_assignable_v<T>);

  static_assert(!cuda::std::move_constructible<T>);
  static_assert(!cuda::std::is_move_assignable_v<T>);
}

TEST_FUNC void test()
{
  {
    static_assert(noexcept(cuda::std::dynamic_format(cuda::std::string_view{})));
    auto format_string = cuda::std::dynamic_format(cuda::std::string_view{});

    using FormatString = decltype(format_string);
    static_assert(cuda::std::same_as<FormatString, cuda::std::__dynamic_format_string<char>>);
    test_properties<FormatString, char>();

#if _CCCL_HAS_WCHAR_T()
    static_assert(noexcept(cuda::std::dynamic_format(cuda::std::wstring_view{})));
    auto wformat_string = cuda::std::dynamic_format(cuda::std::wstring_view{});

    using WFormatString = decltype(wformat_string);
    static_assert(cuda::std::same_as<WFormatString, cuda::std::__dynamic_format_string<wchar_t>>);
    test_properties<WFormatString, wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
  }
}

int main(int, char**)
{
  test();

  return 0;
}
