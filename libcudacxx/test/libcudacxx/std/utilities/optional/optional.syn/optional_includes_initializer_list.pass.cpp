//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// #include <cuda/std/initializer_list>

#include <cuda/std/optional>

#include "test_macros.h"

int main(int, char**)
{
  using cuda::std::optional;

  cuda::std::initializer_list<int> list;
  unused(list);

  return 0;
}
