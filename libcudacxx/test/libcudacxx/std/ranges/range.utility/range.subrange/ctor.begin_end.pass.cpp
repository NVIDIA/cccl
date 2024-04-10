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

// convertible-to-non-slicing cases:
//   1. Not convertible (fail)
//   2. Only one is a pointer (succeed)
//   3. Both are not pointers (succeed)
//   4. Pointer elements are different types (fail)
//   5. Pointer elements are same type (succeed)

// !StoreSize ctor.
static_assert(cuda::std::is_constructible_v<ForwardSubrange, ForwardIter, ForwardIter>); // Default case.
static_assert(!cuda::std::is_constructible_v<ForwardSubrange, Empty, ForwardIter>); // 1.
static_assert(cuda::std::is_constructible_v<ConvertibleForwardSubrange, ConvertibleForwardIter, int*>); // 2.
static_assert(cuda::std::is_constructible_v<ForwardSubrange, ForwardIter, ForwardIter>); // 3. (Same as default case.)
// 4. and 5. must be sized.

__host__ __device__ constexpr bool test()
{
  ForwardSubrange a(ForwardIter(globalBuff), ForwardIter(globalBuff + 8));
  assert(base(a.begin()) == globalBuff);
  assert(base(a.end()) == globalBuff + 8);

  ConvertibleForwardSubrange b(ConvertibleForwardIter(globalBuff), globalBuff + 8);
  assert(b.begin() == globalBuff);
  assert(b.end() == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
