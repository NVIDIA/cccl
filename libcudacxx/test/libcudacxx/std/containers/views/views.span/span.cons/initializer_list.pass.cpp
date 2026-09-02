//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// template <class _InitListValueType>
//   requires same_as<_InitListValueType, value_type> && is_const_v<element_type>
// constexpr span(initializer_list<_InitListValueType> __il);

#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/span>

#include "test_macros.h"

struct A
{};

// Test dynamic extent construction from braced init list for various types.
// Only check size — the initializer_list temporary is destroyed after construction,
// so accessing elements through the span would be dangling.
template <typename T>
TEST_FUNC constexpr bool testDynamicExtent()
{
  cuda::std::span<const T> s1{T{}};
  cuda::std::span<const T> s2{T{}, T{}};
  cuda::std::span<const T> s3{T{}, T{}, T{}};
  cuda::std::span<const T> s4{T{}, T{}, T{}, T{}};
  return s1.size() == 1 && s2.size() == 2 && s3.size() == 3 && s4.size() == 4;
}

// Test static extent construction from braced init list.
// Only check size — same dangling concern as testDynamicExtent.
template <typename T>
TEST_FUNC constexpr bool testStaticExtent()
{
  cuda::std::span<const T, 3> s{T{}, T{}, T{}};
  return s.size() == 3;
}

// Test construction from an explicit initializer_list variable
TEST_FUNC constexpr bool testFromInitializerListVariable()
{
  cuda::std::initializer_list<int> il = {1, 2, 3, 4};
  cuda::std::span<const int> s{il};
  return s.data() == il.begin() && s.size() == il.size();
}

// Test construction with static extent from an explicit initializer_list variable
TEST_FUNC constexpr bool testFromInitializerListVariableStaticExtent()
{
  cuda::std::initializer_list<int> il = {10, 20, 30};
  cuda::std::span<const int, 3> s{il};
  return s.data() == il.begin() && s.size() == il.size();
}

// Test empty initializer_list with dynamic extent
TEST_FUNC constexpr bool testEmptyDynamic()
{
  cuda::std::span<const int> s{cuda::std::initializer_list<int>{}};
  return s.size() == 0;
}

// Test empty initializer_list with static extent 0
TEST_FUNC constexpr bool testEmptyStatic()
{
  cuda::std::span<const int, 0> s{cuda::std::initializer_list<int>{}};
  return s.size() == 0;
}

// Test value integrity: verify size and element values
// Must use a named initializer_list so the backing storage outlives the span.
TEST_FUNC constexpr bool testValueIntegrity()
{
  cuda::std::initializer_list<int> il = {1, 2, 3, 4, 5};
  cuda::std::span<const int> s{il};
  return s.size() == il.size() && s[0] == 1 && s[1] == 2 && s[2] == 3 && s[3] == 4 && s[4] == 5;
}

// Test with bool literals
TEST_FUNC constexpr bool testBool()
{
  cuda::std::initializer_list<bool> il = {true, false, true};
  cuda::std::span<const bool> s{il};
  return s.size() == il.size() && s[0] && !s[1] && s[2];
}

// Test const volatile element type (only check size, volatile reads are not constexpr)
TEST_FUNC constexpr bool testConstVolatile()
{
  cuda::std::initializer_list<int> il = {1, 2, 3};
  cuda::std::span<const volatile int> s{il};
  return s.size() == il.size();
}

// Test const pointer element type: span<int* const> from initializer_list<int*>.
// is_const_v<int* const> is true (the pointer itself is const), so this should work.
TEST_FUNC bool testConstPointer()
{
  int x                                = 1;
  int y                                = 2;
  cuda::std::initializer_list<int*> il = {&x, &y};
  cuda::std::span<int* const> s{il};
  return s.size() == il.size() && s[0] == &x && s[1] == &y;
}

TEST_FUNC constexpr bool testAll()
{
  // Dynamic extent
  assert(testDynamicExtent<int>());
  assert(testDynamicExtent<A>());

  // Static extent
  assert(testStaticExtent<int>());
  assert(testStaticExtent<A>());

  // From explicit initializer_list variable
  assert(testFromInitializerListVariable());
  assert(testFromInitializerListVariableStaticExtent());

  // Empty initializer_list
  assert(testEmptyDynamic());
  assert(testEmptyStatic());

  // Value integrity
  assert(testValueIntegrity());

  // Bool
  assert(testBool());

  // const volatile
  assert(testConstVolatile());

  return true;
}

// Separate from testAll because testConstPointer uses address-of, which is not constexpr
TEST_FUNC bool testRuntime()
{
  assert(testConstPointer());
  return true;
}

int main(int, char**)
{
  testAll();
  static_assert(testAll());
  testRuntime();

  return 0;
}
