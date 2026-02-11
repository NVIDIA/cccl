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

  __host__ __device__ constexpr friend bool operator==(const ComparisonObject& lhs, const ComparisonObject& rhs) noexcept
  {
    return lhs.value == rhs.value;
  }
};

// Test suit for numeric types
__host__ __device__ constexpr bool test_numeric_types()
{
  // integral values
  {
    constexpr cuda::__equal_to_value<int> eq(1);
    static_assert(eq(1) == true, "");
    static_assert(eq(2) == false, "");
  }

  // floating point values
  {
    constexpr cuda::__equal_to_value<double> eq(3.14);
    static_assert(eq(3.14) == true, "");
    static_assert(eq(2.71) == false, "");
  }

  // default value
  {
    constexpr cuda::__equal_to_value<int> eq;
    static_assert(eq(0) == true, "");
    static_assert(eq(1) == false, "");
  }

  return true;
}

// Test suit for heterogeneous comparisons
__host__ __device__ constexpr bool test_heterogeneous_comparisons()
{
  constexpr cuda::__equal_to_value<int> eq(42);
  static_assert(eq(42.0) == true, "");
  static_assert(eq(43.0) == false, "");

  constexpr cuda::__equal_to_value<double> eqd(42.0);
  static_assert(eqd(42) == true, "");
  static_assert(eqd(43) == false, "");

  return true;
}

// Test suit for user-defined types
__host__ __device__ constexpr bool test_user_defined_types()
{
  constexpr ComparisonObject a{42};
  constexpr ComparisonObject b{42};
  constexpr ComparisonObject c{43};

  constexpr cuda::__equal_to_value<ComparisonObject> eq(a);
  static_assert(eq(b) == true, "");
  static_assert(eq(c) == false, "");

  return true;
}

// Test suit for CTAD
__host__ __device__ constexpr bool test_ctad()
{
  // built-in types
  constexpr cuda::__equal_to_value eq(42);
  static_assert(cuda::std::is_same_v<decltype(eq), const cuda::__equal_to_value<int>>, "");
  static_assert(eq(42) == true, "");
  static_assert(eq(43) == false, "");

  // user-defined types
  constexpr ComparisonObject obj{42};
  constexpr cuda::__equal_to_value eq_obj(obj);
  static_assert(cuda::std::is_same_v<decltype(eq_obj), const cuda::__equal_to_value<ComparisonObject>>, "");
  static_assert(eq_obj(obj) == true, "");
  static_assert(eq_obj(ComparisonObject{43}) == false, "");

  return true;
}

// Test suit for noexcept
__host__ __device__ constexpr bool test_noexcept()
{
  // built-in types
  constexpr cuda::__equal_to_value<int> eq(42);
  static_assert(noexcept(eq(42)) == true, "");
  static_assert(eq(42) == true, "");

  // user-defined types
  constexpr ComparisonObject obj{42};
  constexpr cuda::__equal_to_value<ComparisonObject> eq_obj(obj);
  static_assert(noexcept(eq_obj(obj)) == true, "");
  static_assert(eq_obj(obj) == true, "");

  return true;
}

// Run all test suits
__host__ __device__ constexpr bool test()
{
  return test_numeric_types() && test_heterogeneous_comparisons() && test_user_defined_types() && test_ctad()
      && test_noexcept();
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
