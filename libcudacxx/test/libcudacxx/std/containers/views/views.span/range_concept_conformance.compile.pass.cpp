//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// span

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/span>

using range = cuda::std::span<int>;

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range>, range::iterator>);
static_assert(cuda::std::ranges::common_range<range>);
static_assert(cuda::std::ranges::random_access_range<range>);
static_assert(cuda::std::ranges::contiguous_range<range>);
static_assert(cuda::std::ranges::view<range> && cuda::std::ranges::enable_view<range>);
static_assert(cuda::std::ranges::sized_range<range>);
static_assert(cuda::std::ranges::borrowed_range<range>);
static_assert(cuda::std::ranges::viewable_range<range>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<range const>, range::iterator>);
static_assert(cuda::std::ranges::common_range<range const>);
static_assert(cuda::std::ranges::random_access_range<range const>);
static_assert(cuda::std::ranges::contiguous_range<range const>);
static_assert(!cuda::std::ranges::view<range const> && !cuda::std::ranges::enable_view<range const>);
static_assert(cuda::std::ranges::sized_range<range const>);
static_assert(cuda::std::ranges::borrowed_range<range const>);
static_assert(cuda::std::ranges::viewable_range<range const>);

int main(int, char**)
{
  return 0;
}
