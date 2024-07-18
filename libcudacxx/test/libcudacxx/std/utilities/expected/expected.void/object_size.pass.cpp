//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/expected>

#include <cuda/std/expected>

#include "test_macros.h"

using cuda::std::expected;

int main(int, char**)
{
  static_assert(sizeof(expected<char, char>) == 2 * sizeof(char), "");
  static_assert(sizeof(expected<short, short>) == 2 * sizeof(short), "");
  static_assert(sizeof(expected<int, int>) == 2 * sizeof(int), "");
  static_assert(sizeof(expected<long long, long long>) == 2 * sizeof(long long), "");

  static_assert(sizeof(expected<long long, char>) == 2 * sizeof(long long), "");
  static_assert(sizeof(expected<long long, short>) == 2 * sizeof(long long), "");
  static_assert(sizeof(expected<long long, int>) == 2 * sizeof(long long), "");
  static_assert(sizeof(expected<long long, long long>) == 2 * sizeof(long long), "");

  static_assert(sizeof(expected<void, char>) == 2 * sizeof(char), "");
  static_assert(sizeof(expected<void, short>) == 2 * sizeof(short), "");
  static_assert(sizeof(expected<void, int>) == 2 * sizeof(int), "");
  static_assert(sizeof(expected<void, long long>) == 2 * sizeof(long long), "");
  return 0;
}
