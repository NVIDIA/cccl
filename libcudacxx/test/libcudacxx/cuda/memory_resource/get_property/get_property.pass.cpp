//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::get_property

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

struct prop_with_value
{
  using value_type = int;
};
struct prop
{};

struct upstream_with_valueless_property
{
  __host__ __device__ friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};

struct upstream_with_stateful_property
{
  __host__ __device__ friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value)
  {
    return 42;
  }
};

struct upstream_with_both_properties
{
  __host__ __device__ friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  __host__ __device__ friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value)
  {
    return 42;
  }
};

__host__ __device__ constexpr bool test()
{
  upstream_with_valueless_property with_valueless{};
  get_property(with_valueless, prop{});

  upstream_with_stateful_property with_value{};
  assert(get_property(with_value, prop_with_value{}) == 42);

  upstream_with_both_properties with_both{};
  get_property(with_both, prop{});
  assert(get_property(with_both, prop_with_value{}) == 42);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
