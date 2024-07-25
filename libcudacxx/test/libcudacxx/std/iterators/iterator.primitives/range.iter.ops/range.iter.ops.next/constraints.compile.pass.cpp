//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// ranges::next
// Make sure we're SFINAE-friendly when the template argument constraints are not met.

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"

#if TEST_STD_VER > 2017
template <class... Args>
concept has_ranges_next = requires(Args&&... args) {
  { cuda::std::ranges::next(cuda::std::forward<Args>(args)...) };
};
#else
template <class... Args>
constexpr bool has_ranges_next = cuda::std::invocable<cuda::std::ranges::__next::__fn, Args...>;
#endif

class not_incrementable
{};
static_assert(!has_ranges_next<not_incrementable>);
static_assert(!has_ranges_next<not_incrementable, cuda::std::ptrdiff_t>);
static_assert(!has_ranges_next<not_incrementable, not_incrementable>);
static_assert(!has_ranges_next<not_incrementable, cuda::std::ptrdiff_t, not_incrementable>);

// Test the test
using It2 = forward_iterator<int*>;
static_assert(has_ranges_next<It2>);
static_assert(has_ranges_next<It2, cuda::std::ptrdiff_t>);
static_assert(has_ranges_next<It2, cuda::std::ptrdiff_t, It2>);

int main(int, char**)
{
  return 0;
}
