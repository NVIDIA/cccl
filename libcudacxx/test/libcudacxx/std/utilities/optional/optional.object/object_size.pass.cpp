//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

#include <cuda/std/optional>

#include "test_macros.h"

using cuda::std::optional;

int main(int, char**)
{
  static_assert(sizeof(optional<char>) == 2 * sizeof(char), "");
  static_assert(sizeof(optional<short>) == 2 * sizeof(short), "");
  static_assert(sizeof(optional<int>) == 2 * sizeof(int), "");
  static_assert(sizeof(optional<long long>) == 2 * sizeof(long long), "");

  static_assert(sizeof(optional<int&>) == sizeof(void*), "");
  return 0;
}
