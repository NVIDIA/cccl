//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class In, class Out>
// concept indirectly_movable_storable;

#include <cuda/std/iterator>

#include "test_macros.h"

template<class I, class O>
  requires cuda::std::indirectly_movable<I, O>
TEST_HOST_DEVICE constexpr bool indirectly_movable_storable_subsumption() {
  return false;
}

template<class I, class O>
  requires cuda::std::indirectly_movable_storable<I, O>
TEST_HOST_DEVICE constexpr bool indirectly_movable_storable_subsumption() {
  return true;
}

static_assert(indirectly_movable_storable_subsumption<int*, int*>());

int main(int, char**)
{
  return 0;
}
