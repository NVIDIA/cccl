//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class T>
//   inline constexpr bool enable_borrowed_range<ref_view<T>> = true;

#include <cuda/std/ranges>

#include "test_range.h"

static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::ref_view<BorrowedRange>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::ref_view<BorrowedView>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::ref_view<NonBorrowedView>>);

int main(int, char**)
{
  return 0;
}
