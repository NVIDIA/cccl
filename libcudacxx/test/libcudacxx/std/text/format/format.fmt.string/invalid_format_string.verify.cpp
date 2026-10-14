//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/__format_>

#include "literal.h"

TEST_FUNC constexpr void test()
{
  // Unmatched '{'
  [[maybe_unused]] constexpr auto f1 = cuda::std::basic_format_string<char>{"{"}; // expected-error

  // Unmatched '}'
  [[maybe_unused]] constexpr auto f2 = cuda::std::basic_format_string<char>{"}"}; // expected-error

  // More placeholders than arguments (automatic indexing)
  [[maybe_unused]] constexpr auto f3 = cuda::std::basic_format_string<char>{"{}"}; // expected-error

  // More placeholders than arguments (manual indexing)
  [[maybe_unused]] constexpr auto f4 = cuda::std::basic_format_string<char>{"{0}"}; // expected-error

  // Argument index out of bounds: only 1 arg provided but index 1 requested
  [[maybe_unused]] constexpr auto f5 = cuda::std::basic_format_string<char, int>{"{1}"}; // expected-error

  // Mixed manual and automatic indexing
  [[maybe_unused]] constexpr auto f6 = cuda::std::basic_format_string<char, int, int>{"{} {1}"}; // expected-error
  [[maybe_unused]] constexpr auto f7 = cuda::std::basic_format_string<char, int, int>{"{0} {}"}; // expected-error

  // Invalid type format specifier for int
  [[maybe_unused]] constexpr auto f8 = cuda::std::basic_format_string<char, int>{"{0:{0}P}"}; // expected-error

  // Precision is not allowed for int
  [[maybe_unused]] constexpr auto f9 = cuda::std::basic_format_string<char, int>{"{.3}"}; // expected-error

  // Bool cannot be used as dynamic width
  [[maybe_unused]] constexpr auto f10 = cuda::std::basic_format_string<char, bool>{"{0:{0}}"}; // expected-error

  // Missing closing brace after format specifier
  [[maybe_unused]] constexpr auto f11 = cuda::std::basic_format_string<char, int>{"{:"}; // expected-error
}

int main(int, char**)
{
  test();
  return 0;
}
