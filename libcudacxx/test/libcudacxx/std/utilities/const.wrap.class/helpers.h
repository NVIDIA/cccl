//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H
#define TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H

#include "test_macros.h"

struct NonStructural
{
  TEST_FUNC constexpr NonStructural(int i)
      : value(i)
  {}

  TEST_FUNC constexpr int get() const
  {
    return value;
  }

private:
  int value;
};

#endif // TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H
