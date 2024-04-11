//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<range _Rp>
// using sentinel_t = decltype(ranges::end(declval<_Rp&>()));

#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<test_range<cpp20_input_iterator>>, sentinel>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<test_range<cpp20_input_iterator> const>, sentinel>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<test_non_const_range<cpp20_input_iterator>>, sentinel>);
static_assert(
  cuda::std::same_as<cuda::std::ranges::sentinel_t<test_common_range<forward_iterator>>, forward_iterator<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<test_common_range<forward_iterator> const>,
                                 forward_iterator<int const*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<test_non_const_common_range<forward_iterator>>,
                                 forward_iterator<int*>>);

int main(int, char**)
{
  return 0;
}
