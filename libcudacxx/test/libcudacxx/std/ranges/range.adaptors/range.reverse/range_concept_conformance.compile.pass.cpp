//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Test that reverse_view conforms to range and view concepts.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(
  cuda::std::ranges::bidirectional_range<cuda::std::ranges::reverse_view<test_view<bidirectional_iterator>>>);
static_assert(
  cuda::std::ranges::random_access_range<cuda::std::ranges::reverse_view<test_view<random_access_iterator>>>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::ranges::reverse_view<test_view<contiguous_iterator>>>);
static_assert(!cuda::std::ranges::contiguous_range<cuda::std::ranges::reverse_view<test_view<contiguous_iterator>>>);

static_assert(cuda::std::ranges::view<cuda::std::ranges::reverse_view<test_view<bidirectional_iterator>>>);

int main(int, char**)
{
  return 0;
}
