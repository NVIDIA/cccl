//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

// <cuda/std/span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il);

// #include <any>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_convertible.h"
#include "test_macros.h"

using cuda::std::is_constructible;

// Constructor constrains
static_assert(is_constructible<cuda::std::span<const int>, cuda::std::initializer_list<int>>::value, "");
static_assert(is_constructible<cuda::std::span<const int, 42>, cuda::std::initializer_list<int>>::value, "");
static_assert(is_constructible<cuda::std::span<const int>, cuda::std::initializer_list<const int>>::value, "");
static_assert(is_constructible<cuda::std::span<const int, 42>, cuda::std::initializer_list<const int>>::value, "");

static_assert(!is_constructible<cuda::std::span<int>, cuda::std::initializer_list<int>>::value, "");
static_assert(!is_constructible<cuda::std::span<int, 42>, cuda::std::initializer_list<int>>::value, "");
static_assert(!is_constructible<cuda::std::span<int>, cuda::std::initializer_list<const int>>::value, "");
static_assert(!is_constructible<cuda::std::span<int, 42>, cuda::std::initializer_list<const int>>::value, "");

// Constructor conditionally explicit

static_assert(!test_convertible<cuda::std::span<const int, 28>, cuda::std::initializer_list<int>>(),
              "This constructor must be explicit");
static_assert(is_constructible<cuda::std::span<const int, 28>, cuda::std::initializer_list<int>>::value, "");
static_assert(test_convertible<cuda::std::span<const int>, cuda::std::initializer_list<int>>(),
              "This constructor must not be explicit");
static_assert(is_constructible<cuda::std::span<const int>, cuda::std::initializer_list<int>>::value, "");

struct Sink
{
  constexpr Sink() = default;
  __host__ __device__ constexpr Sink(Sink*) {}
};

__host__ __device__ constexpr cuda::std::size_t count(cuda::std::span<const Sink> sp)
{
  return sp.size();
}

template <cuda::std::size_t N>
__host__ __device__ constexpr cuda::std::size_t count_n(cuda::std::span<const Sink, N> sp)
{
  return sp.size();
}

__host__ __device__ constexpr bool test()
{
  // Dynamic extent
  {
    Sink a[10]{};

    assert(count({a}) == 1);
    assert(count({a, a + 10}) == 2);
    assert(count({a, a + 1, a + 2}) == 3);
    assert(count(cuda::std::initializer_list<Sink>{a[0], a[1], a[2], a[3]}) == 4);
  }

  return true;
}

// Test P2447R4 "Annex C examples"

__host__ __device__ constexpr int three(cuda::std::span<void* const> sp)
{
  return static_cast<int>(sp.size());
}

__host__ __device__ bool test_P2447R4_annex_c_examples()
{
  // 1. Overload resolution is affected
  // --> tested in "initializer_list.verify.cpp"

  // 2. The `initializer_list` ctor has high precedence
  // --> tested in "initializer_list.verify.cpp"

  // 3. Implicit two-argument construction with a highly convertible value_type
  {
    void* a[10];
    assert(three({a, 0}) == 2);
  }
  // {
  //   cuda::std::any a[10];
  //   assert(four({a, a + 10}) == 2);
  // }

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test(), "");

  assert(test_P2447R4_annex_c_examples());

  return 0;
}
