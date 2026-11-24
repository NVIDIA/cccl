//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H
#define TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H

#include <cuda/std/iterator>

#include "test_macros.h"

struct DefaultConstructibleTo42
{
  int val_;

  TEST_FUNC constexpr DefaultConstructibleTo42(const int val = 42) noexcept
      : val_(val)
  {}

  TEST_FUNC friend constexpr bool operator==(DefaultConstructibleTo42 lhs, DefaultConstructibleTo42 rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend constexpr bool operator!=(DefaultConstructibleTo42 lhs, DefaultConstructibleTo42 rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
};

struct NotDefaultConstructible
{
  int val_;

  TEST_FUNC constexpr NotDefaultConstructible(const int val) noexcept
      : val_(val)
  {}
  TEST_FUNC friend constexpr bool operator==(NotDefaultConstructible lhs, NotDefaultConstructible rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend constexpr bool operator!=(NotDefaultConstructible lhs, NotDefaultConstructible rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
};

#endif // TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H
