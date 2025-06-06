//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

// convertible-to-non-slicing cases:
//   1. Not convertible (fail)
//   2. Only one is a pointer (succeed)
//   3. Both are not pointers (succeed)
//   4. Pointer elements are different types (fail)
//   5. Pointer elements are same type (succeed)

static_assert(cuda::std::is_constructible_v<SizedSentinelForwardSubrange,
                                            ConditionallyConvertibleIter,
                                            ConditionallyConvertibleIter,
                                            ConditionallyConvertibleIter::udifference_type>); // Default case.
static_assert(!cuda::std::is_constructible_v<SizedSentinelForwardSubrange,
                                             Empty,
                                             ConditionallyConvertibleIter,
                                             ConditionallyConvertibleIter::udifference_type>); // 1.
static_assert(cuda::std::is_constructible_v<ConvertibleSizedSentinelForwardSubrange,
                                            ConvertibleSizedSentinelForwardIter,
                                            int*,
                                            ConvertibleSizedSentinelForwardIter::udifference_type>); // 2.
static_assert(cuda::std::is_constructible_v<SizedSentinelForwardSubrange,
                                            ConditionallyConvertibleIter,
                                            ConditionallyConvertibleIter,
                                            ConditionallyConvertibleIter::udifference_type>); // 3. (Same as default
                                                                                              // case.)
static_assert(!cuda::std::is_constructible_v<SizedIntPtrSubrange, long*, int*, size_t>); // 4.
static_assert(cuda::std::is_constructible_v<SizedIntPtrSubrange, int*, int*, size_t>); // 5.

__host__ __device__ constexpr bool test()
{
  SizedSentinelForwardSubrange a(
    ConditionallyConvertibleIter(globalBuff), ConditionallyConvertibleIter(globalBuff + 8), 8);
  assert(a.begin().base() == globalBuff);
  assert(a.end().base() == globalBuff + 8);
  assert(a.size() == 8);

  ConvertibleSizedSentinelForwardSubrange b(
    ConvertibleSizedSentinelForwardIter(globalBuff), ConvertibleSizedSentinelForwardIter(globalBuff + 8), 8);
  assert(b.begin() == globalBuff);
  assert(b.end() == globalBuff + 8);
  assert(b.size() == 8);

  SizedIntPtrSubrange c(globalBuff, globalBuff + 8, 8);
  assert(c.begin() == globalBuff);
  assert(c.end() == globalBuff + 8);
  assert(c.size() == 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
