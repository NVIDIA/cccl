//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc-19.16

// template<class T>
// concept sized_range;

#include <cuda/std/ranges>

#include "test_macros.h"

template <cuda::std::ranges::range R>
TEST_HOST_DEVICE consteval bool check_subsumption() {
  return false;
}

template <cuda::std::ranges::sized_range R>
TEST_HOST_DEVICE consteval bool check_subsumption() {
  return true;
}

static_assert(check_subsumption<int[5]>(), "");

int main(int, char**) {
  return 0;
}
