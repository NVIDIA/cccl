//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): Fix failing tests.

#include <cuda/std/cassert>
#include <cuda/std/string_view>

#include "format_functions_common.h"
#include "test_macros.h"

// Provided by the selected checker.
TEST_FUNC bool check(...);
TEST_FUNC bool check_exception(...);

template <class CharT>
TEST_FUNC void test_buffer_copy()
{
  // *** copy ***
  assert(check(SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
               SV("{}"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
               SV("{}"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
    SV("{}"),
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
    SV("{}"),
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
    SV("{}"),
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  // *** copy + push_back ***

  assert(check(SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "X"),
               SV("{}X"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "X"),
               SV("{}X"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  //   assert(check(
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "X"),
  //     SV("{}X"),
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  //   assert(check(
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "X"),
  //     SV("{}X"),
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  //   assert(check(
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "X"),
  //     SV("{}X"),
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  // ***  push_back + copy ***

  assert(check(SV("X"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
               SV("X{}"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(SV("X"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
               SV("X{}"),
               SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  assert(check(
    SV("X"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
    SV("X{}"),
    SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  //   assert(check(
  //     SV("X"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
  //     SV("X{}"),
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));

  //   assert(check(
  //     SV("X"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
  //     SV("X{}"),
  //     SV("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  //        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")));
}

template <class CharT>
TEST_FUNC void test_buffer_fill()
{
  // *** fill ***
  assert(check(SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"), SV("{:|<64}"), SV("")));

  assert(check(SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
               SV("{:|<128}"),
               SV("")));

  assert(check(SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
               SV("{:|<256}"),
               SV("")));

  assert(check(
    SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
    SV("{:|<512}"),
    SV("")));

  assert(check(
    SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
       "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
    SV("{:|<1024}"),
    SV("")));

  // *** fill + push_back ***

  assert(check(SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "X"),
               SV("{:|<64}X"),
               SV("")));

  assert(check(SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "X"),
               SV("{:|<128}X"),
               SV("")));

  //   assert(check(
  //     SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "X"),
  //     SV("{:|<256}X"),
  //     SV("")));

  //   assert(check(
  //     SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "X"),
  //     SV("{:|<512}X"),
  //     SV("")));

  //   assert(check(
  //     SV("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "X"),
  //     SV("{:|<1024}X"),
  //     SV("")));

  // *** push_back + fill ***

  assert(check(SV("X"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
               SV("X{:|<64}"),
               SV("")));

  assert(check(SV("X"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                  "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
               SV("X{:|<128}"),
               SV("")));

  //   assert(check(
  //     SV("X"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
  //     SV("X{:|<256}"),
  //     SV("")));

  //   assert(check(
  //     SV("X"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
  //     SV("X{:|<512}"),
  //     SV("")));

  //   assert(check(
  //     SV("X"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
  //        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"),
  //     SV("X{:|<1024}"),
  //     SV("")));
}

/// Tests special buffer functions with a "large" input.
///
/// This is a test specific for libc++, however the code should behave the same
/// on all implementations.
/// In \c __fmt_output_buffer there are some special functions to optimize
/// outputting multiple characters, \c __copy, \c __transform, \c __fill. This
/// test validates whether the functions behave properly when the output size
/// doesn't fit in its internal buffer.
template <class CharT>
TEST_FUNC void test_buffer_optimizations()
{
  // Used to validate our test sets are the proper size.
  // To test the chunked operations it needs to be larger than the internal
  // buffer. Picked a nice looking number.
  constexpr int minimum = 3 * 256;

  // Copy
  const auto str = SV(
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog."
    "The quick brown fox jumps over the lazy dog.");
  assert(str.size() > minimum);
  assert(check(cuda::std::basic_string_view<CharT>{str}, SV("{}"), str));

  // todo(dabayer): Make this work.
  // Fill
  // cuda::std::inplace_vector<CharT, minimum> fill(minimum, CharT('*'));
  // check(cuda::std::basic_string_view<CharT>{str + fill}, SV("{:*<{}}"), str, str.size() + minimum);
  // check(cuda::std::basic_string_view<CharT>{fill + str + fill}, SV("{:*^{}}"), str, minimum + str.size() + minimum);
  // check(cuda::std::basic_string_view<CharT>{fill + str}, SV("{:*>{}}"), str, minimum + str.size());
}

TEST_FUNC void test()
{
  test_buffer_copy<char>();
  test_buffer_fill<char>();
  test_buffer_optimizations<char>();
#if _CCCL_HAS_WCHAR_T()
  test_buffer_copy<wchar_t>();
  test_buffer_fill<wchar_t>();
  test_buffer_optimizations<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
