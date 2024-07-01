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

// constexpr explicit sentinel(sentinel_t<Base> end);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

struct Sent
{
  int i;

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
};

// Test explicit

static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<cuda::std::ranges::elements_view<Range, 0>>, Sent>);
static_assert(
  !cuda::std::is_convertible_v<Sent, cuda::std::ranges::sentinel_t<cuda::std::ranges::elements_view<Range, 0>>>);

__host__ __device__ constexpr bool test()
{
  // base is init correctly
  {
    using R        = cuda::std::ranges::elements_view<Range, 0>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    Sentinel s1(Sent{5});
    assert(s1.base().i == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
