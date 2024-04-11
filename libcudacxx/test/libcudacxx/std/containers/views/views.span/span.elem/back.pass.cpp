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

// constexpr reference back() const noexcept;
//   Expects: empty() is false.
//   Effects: Equivalent to: return *(data() + (size() - 1));
//

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <typename Span>
__host__ __device__ constexpr bool testConstexprSpan(Span sp)
{
  ASSERT_NOEXCEPT(sp.back());
  return &sp.back() == sp.data() + (sp.size() - 1);
}

template <typename Span>
__host__ __device__ void testRuntimeSpan(Span sp)
{
  ASSERT_NOEXCEPT(sp.back());
  assert(&sp.back() == sp.data() + (sp.size() - 1));
}

template <typename Span>
__host__ __device__ void testEmptySpan(Span sp)
{
  if (!sp.empty())
  {
    unused(sp.back());
  }
}

struct A
{};
constexpr int iArr1[]              = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
STATIC_TEST_GLOBAL_VAR int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int>(iArr1, 1)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int>(iArr1, 2)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int>(iArr1, 3)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int>(iArr1, 4)));

  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int, 1>(iArr1, 1)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int, 2>(iArr1, 2)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int, 3>(iArr1, 3)));
  STATIC_ASSERT_CXX14(testConstexprSpan(cuda::std::span<const int, 4>(iArr1, 4)));

  testRuntimeSpan(cuda::std::span<int>(iArr2, 1));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 2));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 3));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 4));

  testRuntimeSpan(cuda::std::span<int, 1>(iArr2, 1));
  testRuntimeSpan(cuda::std::span<int, 2>(iArr2, 2));
  testRuntimeSpan(cuda::std::span<int, 3>(iArr2, 3));
  testRuntimeSpan(cuda::std::span<int, 4>(iArr2, 4));

  cuda::std::span<int, 0> sp;
  testEmptySpan(sp);

  return 0;
}
