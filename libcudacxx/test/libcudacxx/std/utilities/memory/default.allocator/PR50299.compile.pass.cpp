//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// Make sure we can use cuda::std::allocator<void> in all Standard modes. While the
// explicit specialization for cuda::std::allocator<void> was deprecated, using that
// specialization was neither deprecated nor removed (in C++20 it should simply
// start using the primary template).
//
// See https://llvm.org/PR50299.

#include <cuda/std/__memory_>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR cuda::std::allocator<void> alloc;

int main(int, char**)
{
  unused(alloc);
  return 0;
}
