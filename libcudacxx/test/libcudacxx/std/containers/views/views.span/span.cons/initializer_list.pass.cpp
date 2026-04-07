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
//
// template <class _Tp>
// span(initializer_list<_Tp>) -> span<const _Tp>;

#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

// Test dynamic extent construction from braced init list for various types
template <typename T>
__host__ __device__ constexpr bool testDynamicExtent()
{
  cuda::std::span<const T> s{T{}, T{}, T{}};
  return s.size() == 3;
}

// Test static extent construction from braced init list
template <typename T>
__host__ __device__ constexpr bool testStaticExtent()
{
  cuda::std::span<const T, 3> s{T{}, T{}, T{}};
  return s.size() == 3;
}

// Test construction from an explicit initializer_list variable
__host__ __device__ constexpr bool testFromInitializerListVariable()
{
  cuda::std::initializer_list<int> il = {1, 2, 3, 4};
  cuda::std::span<const int> s{il};
  return s.data() == il.begin() && s.size() == 4;
}

// Test construction with static extent from an explicit initializer_list variable
__host__ __device__ constexpr bool testFromInitializerListVariableStaticExtent()
{
  cuda::std::initializer_list<int> il = {10, 20, 30};
  cuda::std::span<const int, 3> s{il};
  return s.data() == il.begin() && s.size() == 3;
}

// Test empty initializer_list with dynamic extent
__host__ __device__ constexpr bool testEmptyDynamic()
{
  cuda::std::span<const int> s{cuda::std::initializer_list<int>{}};
  return s.size() == 0;
}

// Test empty initializer_list with static extent 0
__host__ __device__ constexpr bool testEmptyStatic()
{
  cuda::std::span<const int, 0> s{cuda::std::initializer_list<int>{}};
  return s.size() == 0;
}

// Test value integrity: verify size and element values
// Must use a named initializer_list so the backing storage outlives the span.
__host__ __device__ constexpr bool testValueIntegrity()
{
  cuda::std::initializer_list<int> il = {1, 2, 3, 4, 5};
  cuda::std::span<const int> s{il};
  return s.size() == 5 && s[0] == 1 && s[1] == 2 && s[2] == 3 && s[3] == 4 && s[4] == 5;
}

// Test with bool literals
__host__ __device__ constexpr bool testBool()
{
  cuda::std::initializer_list<bool> il = {true, false, true};
  cuda::std::span<const bool> s{il};
  return s.size() == 3 && s[0] == true && s[1] == false && s[2] == true;
}

// Test const volatile element type
__host__ __device__ constexpr bool testConstVolatile()
{
  cuda::std::initializer_list<int> il = {1, 2, 3};
  cuda::std::span<const volatile int> s{il};
  return s.size() == 3 && s[0] == 1 && s[1] == 2 && s[2] == 3;
}

// Test CTAD: span{1, 2, 3} should deduce span<const int, dynamic_extent>
__host__ __device__ constexpr bool testCTAD()
{
  cuda::std::initializer_list<int> il = {1, 2, 3};
  cuda::std::span s{il};
  static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<const int>>, "");
  return s.size() == 3 && s[0] == 1 && s[1] == 2 && s[2] == 3;
}

__host__ __device__ constexpr bool testAll()
{
  // Dynamic extent
  assert(testDynamicExtent<int>());
  assert(testDynamicExtent<long>());
  assert(testDynamicExtent<double>());
  assert(testDynamicExtent<A>());

  // Static extent
  assert(testStaticExtent<int>());
  assert(testStaticExtent<long>());
  assert(testStaticExtent<double>());
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

  // CTAD
  assert(testCTAD());

  return true;
}

int main(int, char**)
{
  testAll();
  static_assert(testAll());

  return 0;
}
