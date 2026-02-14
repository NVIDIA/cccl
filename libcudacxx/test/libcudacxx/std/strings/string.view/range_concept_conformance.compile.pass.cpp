//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/string_view>

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<cuda::std::string_view>, cuda::std::string_view::iterator>);
static_assert(cuda::std::ranges::common_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::contiguous_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::view<cuda::std::string_view>
              && cuda::std::ranges::enable_view<cuda::std::string_view>);
static_assert(cuda::std::ranges::sized_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::viewable_range<cuda::std::string_view>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<cuda::std::string_view const>,
                                 cuda::std::string_view::const_iterator>);
static_assert(cuda::std::ranges::common_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::contiguous_range<cuda::std::string_view const>);
static_assert(!cuda::std::ranges::view<cuda::std::string_view const>
              && !cuda::std::ranges::enable_view<cuda::std::string_view const>);
static_assert(cuda::std::ranges::sized_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::viewable_range<cuda::std::string_view const>);

int main(int, char**)
{
  return 0;
}
