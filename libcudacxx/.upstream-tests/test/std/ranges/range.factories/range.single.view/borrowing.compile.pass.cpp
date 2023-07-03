//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// single_view does not specialize enable_borrowed_range

#include <cuda/std/ranges>

#include "test_range.h"

static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::single_view<int>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::single_view<int*>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::single_view<BorrowedView>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::single_view<NonBorrowedView>>);

int main(int, char**) {
  return 0;
}
