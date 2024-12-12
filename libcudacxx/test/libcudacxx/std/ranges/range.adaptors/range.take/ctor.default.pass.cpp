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

// take_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int buff[8] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructible : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr DefaultConstructible()
      : begin_(buff)
      , end_(buff + 8)
  {}
  __host__ __device__ constexpr int const* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int const* end() const
  {
    return end_;
  }

private:
  int const* begin_;
  int const* end_;
};

struct NonDefaultConstructible : cuda::std::ranges::view_base
{
  NonDefaultConstructible() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_view<DefaultConstructible> tv;
    assert(tv.begin() == buff);
    assert(tv.size() == 0);
  }

  // Test SFINAE-friendliness
  {
    static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<DefaultConstructible>>);
    static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::take_view<NonDefaultConstructible>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
