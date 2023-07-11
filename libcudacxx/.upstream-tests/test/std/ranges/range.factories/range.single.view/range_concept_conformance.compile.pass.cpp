//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Test that single_view conforms to range and view concepts.

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_iterators.h"

struct Empty {};

static_assert(cuda::std::ranges::contiguous_range<cuda::std::ranges::single_view<Empty>>);
static_assert(cuda::std::ranges::contiguous_range<const cuda::std::ranges::single_view<Empty>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::single_view<Empty>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::single_view<const Empty>>);
static_assert(cuda::std::ranges::contiguous_range<const cuda::std::ranges::single_view<const Empty>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::single_view<int>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::single_view<const int>>);
static_assert(cuda::std::ranges::contiguous_range<const cuda::std::ranges::single_view<const int>>);

int main(int, char**) {
  return 0;
}