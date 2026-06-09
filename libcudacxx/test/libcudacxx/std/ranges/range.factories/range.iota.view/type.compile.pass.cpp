//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: variadic function is unsupported in tile code

#include <cuda/std/ranges>

#include "test_macros.h"

// Test that we SFINAE away iota_view<bool>.

template <class T>
TEST_FUNC cuda::std::ranges::iota_view<T> f(int);
template <class T>
TEST_FUNC void f(...)
{}

TEST_FUNC void test()
{
  f<bool>(42);
}

int main(int, char**)
{
  return 0;
}
