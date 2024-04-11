//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

static_assert(cuda::std::is_constructible_v<ForwardSubrange, ForwardBorrowedRange>); // Default case.
static_assert(!cuda::std::is_constructible_v<ForwardSubrange, ForwardRange>); // Not borrowed.
// Iter convertible to sentinel (pointer) type.
static_assert(cuda::std::is_constructible_v<ConvertibleForwardSubrange, ConvertibleForwardBorrowedRange>);
// Where neither iter or sentinel are pointers, but they are different.
static_assert(cuda::std::is_constructible_v<DifferentSentinelSubrange, ForwardBorrowedRangeDifferentSentinel>);
static_assert(cuda::std::is_constructible_v<DifferentSentinelWithSizeMemberSubrange, DifferentSentinelWithSizeMember>);

__host__ __device__ constexpr bool test()
{
  ForwardSubrange a{ForwardBorrowedRange()};
  assert(base(a.begin()) == globalBuff);
  assert(base(a.end()) == globalBuff + 8);

  ConvertibleForwardSubrange b{ConvertibleForwardBorrowedRange()};
  assert(b.begin() == globalBuff);
  assert(b.end() == globalBuff + 8);

  DifferentSentinelSubrange c{ForwardBorrowedRangeDifferentSentinel()};
  assert(base(c.begin()) == globalBuff);
  assert(c.end().value == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
