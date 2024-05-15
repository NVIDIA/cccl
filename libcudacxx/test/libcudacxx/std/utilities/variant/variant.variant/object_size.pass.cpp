//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// <cuda/std/variant>

#include <cuda/std/variant>

#include "test_macros.h"

using cuda::std::variant;

int main(int, char**)
{
  static_assert(sizeof(variant<char>) == 2 * sizeof(char), "");
  static_assert(sizeof(variant<short>) == 2 * sizeof(short), "");
  static_assert(sizeof(variant<int>) == 2 * sizeof(int), "");
  static_assert(sizeof(variant<long long>) == 2 * sizeof(long long), "");

  static_assert(sizeof(variant<char, char>) == 2 * sizeof(char), "");
  static_assert(sizeof(variant<char, short>) == 2 * sizeof(short), "");
  static_assert(sizeof(variant<char, int>) == 2 * sizeof(int), "");
  static_assert(sizeof(variant<char, long long>) == 2 * sizeof(long long), "");
  return 0;
}
