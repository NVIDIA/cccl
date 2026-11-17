//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CLEAR_PADDING_HELPERS_H
#define TEST_CLEAR_PADDING_HELPERS_H

#include <cuda/std/type_traits>
#include <cuda/type_traits>

#include "test_macros.h"

struct WithPadding
{
  int a;
  char b;
};

struct RelaxedWithPadding
{
  int a;
  char b;

  // nontrivial destructor to prevent cuda::std::is_trivially_copyable_v
  TEST_FUNC ~RelaxedWithPadding() {} // NOLINT(modernize-use-equals-default)

  TEST_FUNC RelaxedWithPadding* operator&()
  {
    return nullptr;
  }
};

static_assert(cuda::is_trivially_copyable_v<RelaxedWithPadding>);
static_assert(!cuda::std::is_trivially_copyable_v<RelaxedWithPadding>);

#endif // TEST_CLEAR_PADDING_HELPERS_H
