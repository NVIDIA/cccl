//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// template<class T, class Pred>
//   inline constexpr bool enable_borrowed_range<drop_while_view<T, Pred>> =
//     enable_borrowed_range<T>;

#include <cuda/std/ranges>

#include "test_macros.h"

struct NonBorrowed : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin();
  TEST_FUNC int* end();
};

struct Borrowed : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin();
  TEST_FUNC int* end();
};

struct Pred
{
  TEST_FUNC bool operator()(int) const;
};

template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::drop_while_view<NonBorrowed, Pred>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::drop_while_view<Borrowed, Pred>>);

int main(int, char**)
{
  return 0;
}
