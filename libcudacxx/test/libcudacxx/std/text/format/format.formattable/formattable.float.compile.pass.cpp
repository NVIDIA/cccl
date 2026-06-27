//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class T, class charT>
// concept formattable = ...

#include <cuda/std/__format_>
#include <cuda/std/concepts>

#include "test_macros.h"

template <class T, class CharT>
TEST_FUNC void assert_is_not_formattable()
{
  static_assert(!cuda::std::formattable<T, CharT>);
}

template <class T, class CharT>
TEST_FUNC void assert_is_formattable()
{
  // Only formatters for CharT == char || CharT == wchar_t are enabled for the
  // standard formatters. When CharT is a different type the formatter should
  // be disabled.
  if constexpr (cuda::std::same_as<CharT, char>
#if _CCCL_HAS_WCHAR_T()
                || cuda::std::same_as<CharT, wchar_t>
#endif // _CCCL_HAS_WCHAR_T()
  )
    static_assert(cuda::std::formattable<T, CharT>);
  else
  {
    assert_is_not_formattable<T, CharT>();
  }
}

template <class CharT>
TEST_FUNC void test()
{
  assert_is_formattable<float, CharT>();
  assert_is_formattable<double, CharT>();
#if _CCCL_HAS_LONG_DOUBLE()
  assert_is_formattable<long double, CharT>();
#endif // _CCCL_HAS_LONG_DOUBLE()
}

TEST_FUNC void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();

  test<int>();
}

int main(int, char**)
{
  return 0;
}
