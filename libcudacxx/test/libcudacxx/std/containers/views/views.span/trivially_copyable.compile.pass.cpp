//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// P2251 was voted into C++23, but is supported even in C++20 mode by all vendors.

// <span>

#include <cuda/std/span>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_trivially_copyable<cuda::std::span<int>>::value, "");
static_assert(cuda::std::is_trivially_copyable<cuda::std::span<int, 3>>::value, "");

int main(int, char**)
{
  return 0;
}
