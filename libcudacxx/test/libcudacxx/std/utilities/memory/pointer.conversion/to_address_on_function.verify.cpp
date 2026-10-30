//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> constexpr T* to_address(T* p) noexcept;
//     Mandates: T is not a function type.

#include <cuda/std/memory>

#include "test_macros.h"

TEST_FUNC int f();

TEST_FUNC void test()
{
  (void) cuda::std::to_address(f); // expected-error@*:* {{is a function type}}
}

int main(int, char**)
{
  return 0;
}
