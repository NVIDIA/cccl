//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// constexpr size_type size_bytes() const noexcept;
//
//  Effects: Equivalent to: return size() * sizeof(element_type);

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <typename Span>
__host__ __device__ constexpr bool testConstexprSpan(Span sp, size_t sz)
{
  static_assert(noexcept(sp.size_bytes()));
  return (size_t) sp.size_bytes() == sz * sizeof(typename Span::element_type);
}

template <typename Span>
__host__ __device__ void testRuntimeSpan(Span sp, size_t sz)
{
  static_assert(noexcept(sp.size_bytes()));
  assert((size_t) sp.size_bytes() == sz * sizeof(typename Span::element_type));
}

struct A
{};
constexpr int iArr1[]            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
TEST_GLOBAL_VARIABLE int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
  static_assert(testConstexprSpan(cuda::std::span<int>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<long>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<double>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<A>(), 0), "");

  static_assert(testConstexprSpan(cuda::std::span<int, 0>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<long, 0>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<double, 0>(), 0), "");
  static_assert(testConstexprSpan(cuda::std::span<A, 0>(), 0), "");

  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 1), 1), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 2), 2), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 3), 3), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 4), 4), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 5), 5), "");

  testRuntimeSpan(cuda::std::span<int>(), 0);
  testRuntimeSpan(cuda::std::span<long>(), 0);
  testRuntimeSpan(cuda::std::span<double>(), 0);
  testRuntimeSpan(cuda::std::span<A>(), 0);

  testRuntimeSpan(cuda::std::span<int, 0>(), 0);
  testRuntimeSpan(cuda::std::span<long, 0>(), 0);
  testRuntimeSpan(cuda::std::span<double, 0>(), 0);
  testRuntimeSpan(cuda::std::span<A, 0>(), 0);

  testRuntimeSpan(cuda::std::span<int>(iArr2, 1), 1);
  testRuntimeSpan(cuda::std::span<int>(iArr2, 2), 2);
  testRuntimeSpan(cuda::std::span<int>(iArr2, 3), 3);
  testRuntimeSpan(cuda::std::span<int>(iArr2, 4), 4);
  testRuntimeSpan(cuda::std::span<int>(iArr2, 5), 5);

  testRuntimeSpan(cuda::std::span<int, 1>(iArr2 + 5, 1), 1);
  testRuntimeSpan(cuda::std::span<int, 2>(iArr2 + 4, 2), 2);
  testRuntimeSpan(cuda::std::span<int, 3>(iArr2 + 3, 3), 3);
  testRuntimeSpan(cuda::std::span<int, 4>(iArr2 + 2, 4), 4);
  testRuntimeSpan(cuda::std::span<int, 5>(iArr2 + 1, 5), 5);

  return 0;
}
