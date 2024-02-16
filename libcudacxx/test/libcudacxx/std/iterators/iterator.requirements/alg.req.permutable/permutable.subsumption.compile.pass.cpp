//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I>
//   concept permutable = see below; // Since C++20

#include <cuda/std/iterator>

#include "test_macros.h"

template<class I> TEST_HOST_DEVICE void test_subsumption() requires cuda::std::forward_iterator<I>;
template<class I> TEST_HOST_DEVICE void test_subsumption() requires cuda::std::indirectly_movable_storable<I, I>;
template<class I> TEST_HOST_DEVICE void test_subsumption() requires cuda::std::indirectly_swappable<I, I>;
template<class I> TEST_HOST_DEVICE constexpr bool test_subsumption() requires cuda::std::permutable<I> { return true; }
static_assert(test_subsumption<int*>());

int main(int, char**)
{
  return 0;
}
