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

// template<class T>
// using iterator_t = decltype(ranges::begin(declval<T&>()));

#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_range.h"

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator>>, cpp17_input_iterator<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_common_range<cpp17_input_iterator> const>,
                                 cpp17_input_iterator<int const*>>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<test_non_const_common_range<cpp17_input_iterator>>,
                                 cpp17_input_iterator<int*>>);

int main(int, char**)
{
  return 0;
}
