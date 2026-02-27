//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/functional>

// equal_to_value

#include <cuda/functional>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

// Dummy comparable type
struct ComparisonObject
{
  int value;

  __host__ __device__ constexpr friend bool operator==(const ComparisonObject& lhs, const ComparisonObject& rhs)
  {
    return lhs.value == rhs.value;
  }
};

// Test suit for numeric types
__host__ __device__ constexpr void test_numeric_types()
{
  // integral values
  {
    const cuda::equal_to_value<int> eq{1};
    assert(eq(1) == true);
    assert(eq(2) == false);
    static_assert(noexcept(eq(1)));
  }

  // floating point values
  {
    const cuda::equal_to_value<double> eq{3.14};
    assert(eq(3.14) == true);
    assert(eq(2.71) == false);
    static_assert(noexcept(eq(3.14)));
  }
}

// Test suit for heterogeneous comparisons
__host__ __device__ constexpr void test_heterogeneous_comparisons()
{
  const cuda::equal_to_value<int> eq{42};
  assert(eq(42.0) == true);
  assert(eq(43.0) == false);
  static_assert(noexcept(eq(42.0)));

  const cuda::equal_to_value<double> eqd{42.0};
  assert(eqd(42) == true);
  assert(eqd(43) == false);
  static_assert(noexcept(eq(42)));
}

// Test suit for user-defined types
__host__ __device__ constexpr void test_user_defined_types()
{
  const ComparisonObject a{42};
  const ComparisonObject b{42};
  const ComparisonObject c{43};

  const cuda::equal_to_value<ComparisonObject> eq{a};
  assert(eq(b) == true);
  assert(eq(c) == false);
  static_assert(!noexcept(eq(b)));
}

// Test suit for CTAD
__host__ __device__ constexpr void test_ctad()
{
  // built-in types
  const cuda::equal_to_value eq{42};
  static_assert(::cuda::std::is_same_v<decltype(eq), const cuda::equal_to_value<int>>);
  assert(eq(42) == true);
  assert(eq(43) == false);

  // user-defined types
  const ComparisonObject obj{42};
  const cuda::equal_to_value eq_obj{obj};
  static_assert(::cuda::std::is_same_v<decltype(eq_obj), const cuda::equal_to_value<ComparisonObject>>);
  assert(eq_obj(obj) == true);
  assert(eq_obj(ComparisonObject{43}) == false);
}

// Run all test suits
__host__ __device__ constexpr bool test()
{
  test_numeric_types();
  test_heterogeneous_comparisons();
  test_user_defined_types();
  test_ctad();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
