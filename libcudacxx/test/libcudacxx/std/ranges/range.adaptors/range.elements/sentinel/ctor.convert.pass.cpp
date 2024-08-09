//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct Sent
{
  int i;
  Sent() = default;
  __host__ __device__ constexpr Sent(int ii)
      : i(ii)
  {}
  __host__ __device__ friend constexpr bool operator==(cuda::std::tuple<int>*, const Sent&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const Sent&, cuda::std::tuple<int>*)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(cuda::std::tuple<int>*, const Sent&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

struct ConstSent
{
  int i;
  ConstSent() = default;
  __host__ __device__ constexpr ConstSent(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr ConstSent(const Sent& s)
      : i(s.i)
  {}
  __host__ __device__ friend constexpr bool operator==(cuda::std::tuple<int>*, const ConstSent&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const ConstSent&, cuda::std::tuple<int>*)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(cuda::std::tuple<int>*, const ConstSent&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const ConstSent&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ cuda::std::tuple<int>* begin() const
  {
    return nullptr;
  }
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
  __host__ __device__ ConstSent end() const
  {
    return ConstSent{};
  }
};

struct NonConvertConstSent
{
  int i;
  NonConvertConstSent() = default;
  __host__ __device__ constexpr NonConvertConstSent(int ii)
      : i(ii)
  {}
  __host__ __device__ friend constexpr bool operator==(cuda::std::tuple<int>*, const NonConvertConstSent&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const NonConvertConstSent&, cuda::std::tuple<int>*)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(cuda::std::tuple<int>*, const NonConvertConstSent&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const NonConvertConstSent&, cuda::std::tuple<int>*)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

struct NonConvertConstSentRange : cuda::std::ranges::view_base
{
  __host__ __device__ cuda::std::tuple<int>* begin() const
  {
    return nullptr;
  }
  __host__ __device__ Sent end()
  {
    return Sent{};
  }
  __host__ __device__ NonConvertConstSent end() const
  {
    return NonConvertConstSent{};
  }
};

// Test Constraint
static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<const cuda::std::ranges::elements_view<Range, 0>>,
                                cuda::std::ranges::sentinel_t<cuda::std::ranges::elements_view<Range, 0>>>);

// !Const
static_assert(
  !cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<cuda::std::ranges::elements_view<Range, 0>>,
                                 cuda::std::ranges::sentinel_t<const cuda::std::ranges::elements_view<Range, 0>>>);

// !convertible_to<sentinel_t<V>, sentinel_t<Base>>
static_assert(!cuda::std::is_constructible_v<
              cuda::std::ranges::sentinel_t<const cuda::std::ranges::elements_view<NonConvertConstSentRange, 0>>,
              cuda::std::ranges::sentinel_t<cuda::std::ranges::elements_view<NonConvertConstSentRange, 0>>>);

__host__ __device__ constexpr bool test()
{
  // base is init correctly
  {
    using R             = cuda::std::ranges::elements_view<Range, 0>;
    using Sentinel      = cuda::std::ranges::sentinel_t<R>;
    using ConstSentinel = cuda::std::ranges::sentinel_t<const R>;
    static_assert(!cuda::std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5});
    ConstSentinel s2 = s1;
    assert(s2.base().i == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
