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

// template<input_or_output_iterator I, sentinel_for<I> S>
//   requires (!same_as<I, S> && copyable<I>)

#include <cuda/std/iterator>

#include "test_iterators.h"

#if TEST_STD_VER >= 2020
template <class I, class S>
concept ValidCommonIterator = requires { typename cuda::std::common_iterator<I, S>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class I, class S, class = void>
inline constexpr bool ValidCommonIterator = false;

template <class I, class S>
inline constexpr bool ValidCommonIterator<I, S, cuda::std::void_t<cuda::std::common_iterator<I, S>>> = true;
#endif // TEST_STD_VER <= 2017

static_assert(ValidCommonIterator<int*, const int*>);
static_assert(!ValidCommonIterator<int, int>); // !input_or_output_iterator<I>
static_assert(!ValidCommonIterator<int*, float*>); // !sentinel_for<S, I>
static_assert(!ValidCommonIterator<int*, int*>); // !same_as<I, S>
static_assert(
  !ValidCommonIterator<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>); // !copyable<I>

int main(int, char**)
{
  return 0;
}
