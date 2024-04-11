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

// <cuda/std/span>

// template<class ElementType, size_t Extent>
// inline constexpr bool ranges::enable_borrowed_range<
//     span<ElementType, Extent>> = true;

#include <cuda/std/span>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::span<int, 0>>);
  static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::span<int, 42>>);
  static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::span<int, cuda::std::dynamic_extent>>);
  static_assert(!cuda::std::ranges::enable_borrowed_range<cuda::std::span<int, 42>&>);
  static_assert(!cuda::std::ranges::enable_borrowed_range<cuda::std::span<int, 42> const>);

  return 0;
}
