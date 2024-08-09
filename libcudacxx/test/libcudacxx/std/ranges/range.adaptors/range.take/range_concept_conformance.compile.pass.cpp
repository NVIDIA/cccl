//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Range concept conformance tests for take_view.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(cuda::std::ranges::input_range<cuda::std::ranges::take_view<test_view<cpp20_input_iterator>>>);
static_assert(cuda::std::ranges::forward_range<cuda::std::ranges::take_view<test_view<forward_iterator>>>);
static_assert(cuda::std::ranges::bidirectional_range<cuda::std::ranges::take_view<test_view<bidirectional_iterator>>>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::ranges::take_view<test_view<random_access_iterator>>>);
static_assert(cuda::std::ranges::contiguous_range<cuda::std::ranges::take_view<test_view<contiguous_iterator>>>);

int main(int, char**)
{
  return 0;
}
