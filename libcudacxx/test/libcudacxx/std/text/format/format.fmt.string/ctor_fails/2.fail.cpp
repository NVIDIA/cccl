//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// template<class T> consteval basic_format_string(const T& s);
//
// This constructor does the compile-time format string validation for the
// std::format* functions.

#include <cuda/std/__format_>

#include "literal.h"

template <class CharT>
__host__ __device__ void test_constructor()
{
  [[maybe_unused]] constexpr cuda::std::basic_format_string<CharT, int> fmt{TEST_STRLIT(CharT, "{0:{0}P}")};
}

__host__ __device__ void test()
{
  test_constructor<char>();
#if _CCCL_HAS_WCHAR_T()
  test_constructor<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
