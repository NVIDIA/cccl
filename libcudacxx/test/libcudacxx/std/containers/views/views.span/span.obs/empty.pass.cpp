//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// [[nodiscard]] constexpr bool empty() const noexcept;
//

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

struct A
{};
constexpr int iArr1[]              = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
STATIC_TEST_GLOBAL_VAR int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
  static_assert(noexcept(cuda::std::span<int>().empty()), "");
  static_assert(noexcept(cuda::std::span<int, 0>().empty()), "");

  static_assert(cuda::std::span<int>().empty(), "");
  static_assert(cuda::std::span<long>().empty(), "");
  static_assert(cuda::std::span<double>().empty(), "");
  static_assert(cuda::std::span<A>().empty(), "");

  static_assert(cuda::std::span<int, 0>().empty(), "");
  static_assert(cuda::std::span<long, 0>().empty(), "");
  static_assert(cuda::std::span<double, 0>().empty(), "");
  static_assert(cuda::std::span<A, 0>().empty(), "");

  static_assert(!cuda::std::span<const int>(iArr1, 1).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 2).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 3).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 4).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 5).empty(), "");

  assert((cuda::std::span<int>().empty()));
  assert((cuda::std::span<long>().empty()));
  assert((cuda::std::span<double>().empty()));
  assert((cuda::std::span<A>().empty()));

  assert((cuda::std::span<int, 0>().empty()));
  assert((cuda::std::span<long, 0>().empty()));
  assert((cuda::std::span<double, 0>().empty()));
  assert((cuda::std::span<A, 0>().empty()));

  assert(!(cuda::std::span<int, 1>(iArr2, 1).empty()));
  assert(!(cuda::std::span<int, 2>(iArr2, 2).empty()));
  assert(!(cuda::std::span<int, 3>(iArr2, 3).empty()));
  assert(!(cuda::std::span<int, 4>(iArr2, 4).empty()));
  assert(!(cuda::std::span<int, 5>(iArr2, 5).empty()));

  return 0;
}
