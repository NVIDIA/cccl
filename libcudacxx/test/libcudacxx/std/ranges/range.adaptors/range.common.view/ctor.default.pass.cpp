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

// common_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "types.h"

int main(int, char**)
{
  static_assert(!cuda::std::default_initializable<cuda::std::ranges::common_view<MoveOnlyView>>);
  static_assert(cuda::std::default_initializable<cuda::std::ranges::common_view<DefaultConstructibleView>>);

  cuda::std::ranges::common_view<DefaultConstructibleView> common;
  assert(common.begin() == static_cast<int*>(nullptr));
  assert(common.end() == static_cast<int*>(nullptr));

  return 0;
}
