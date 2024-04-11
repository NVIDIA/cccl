//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// array

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

using range = cuda::std::array<int, 10>;

static_assert(!cuda::std::ranges::view<range>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range>, range::iterator>);
static_assert(cuda::std::ranges::common_range<range>);
static_assert(cuda::std::ranges::random_access_range<range>);
static_assert(cuda::std::ranges::contiguous_range<range>);
static_assert(cuda::std::ranges::sized_range<range>);
static_assert(!cuda::std::ranges::borrowed_range<range>);
static_assert(cuda::std::ranges::viewable_range<range>);

static_assert(!cuda::std::ranges::view<range const>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range const>, range::const_iterator>);
static_assert(cuda::std::ranges::common_range<range const>);
static_assert(cuda::std::ranges::random_access_range<range const>);
static_assert(cuda::std::ranges::contiguous_range<range const>);
static_assert(cuda::std::ranges::sized_range<range const>);
static_assert(!cuda::std::ranges::borrowed_range<range const>);
static_assert(!cuda::std::ranges::viewable_range<range const>);

int main(int, char**)
{
  return 0;
}
