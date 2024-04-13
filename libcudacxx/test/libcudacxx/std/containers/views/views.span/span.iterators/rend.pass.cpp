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

// constexpr       reverse_iterator  rend() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <typename Span>
__host__ __device__ constexpr bool testConstexprSpan(Span s)
{
  bool ret                          = true;
  typename Span::reverse_iterator e = s.rend();
  if (s.empty())
  {
    ret = ret && (e == s.rbegin());
  }
  else
  {
    ret = ret && (e != s.rbegin());
  }

  ret = ret && (static_cast<size_t>(e - s.rbegin()) == s.size());
  return ret;
}

template <class Span>
__host__ __device__ void testRuntimeSpan(Span s)
{
  typename Span::reverse_iterator e = s.rend();
  if (s.empty())
  {
    assert(e == s.rbegin());
  }
  else
  {
    assert(e != s.rbegin());
  }

  assert(static_cast<size_t>(e - s.rbegin()) == s.size());
}

struct A
{};
__host__ __device__ bool operator==(A, A)
{
  return true;
}

#if TEST_STD_VER > 2014
constexpr int iArr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#endif
STATIC_TEST_GLOBAL_VAR int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
#if TEST_STD_VER > 2014
  static_assert(testConstexprSpan(cuda::std::span<int>()), "");
  static_assert(testConstexprSpan(cuda::std::span<long>()), "");
  static_assert(testConstexprSpan(cuda::std::span<double>()), "");
  static_assert(testConstexprSpan(cuda::std::span<A>()), "");

  static_assert(testConstexprSpan(cuda::std::span<int, 0>()), "");
  static_assert(testConstexprSpan(cuda::std::span<long, 0>()), "");
  static_assert(testConstexprSpan(cuda::std::span<double, 0>()), "");
  static_assert(testConstexprSpan(cuda::std::span<A, 0>()), "");

  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 1)), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 2)), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 3)), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 4)), "");
  static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 5)), "");
#endif // TEST_STD_VER > 2014

  testRuntimeSpan(cuda::std::span<int>());
  testRuntimeSpan(cuda::std::span<long>());
  testRuntimeSpan(cuda::std::span<double>());
  testRuntimeSpan(cuda::std::span<A>());

  testRuntimeSpan(cuda::std::span<int, 0>());
  testRuntimeSpan(cuda::std::span<long, 0>());
  testRuntimeSpan(cuda::std::span<double, 0>());
  testRuntimeSpan(cuda::std::span<A, 0>());

  testRuntimeSpan(cuda::std::span<int>(iArr2, 1));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 2));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 3));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 4));
  testRuntimeSpan(cuda::std::span<int>(iArr2, 5));

  return 0;
}
