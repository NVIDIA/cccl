//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// cuda::get_property
#include <cuda/std/cassert>
#include <cuda/memory_resource>

struct prop_with_value {
  using value_type = int;
};
struct prop {};

struct upstream_with_valueless_property {
  friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_valueless_property, prop>, "");
static_assert(!cuda::std::invocable<decltype(cuda::get_property), upstream_with_valueless_property, prop_with_value>, "");

struct upstream_with_stateful_property {
  friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value) {
    return 42;
  }
};
static_assert(!cuda::std::invocable<decltype(cuda::get_property), upstream_with_stateful_property, prop>, "");
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_stateful_property, prop_with_value>, "");

struct upstream_with_both_properties {
  friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value) {
    return 42;
  }
};
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_both_properties, prop>, "");
static_assert( cuda::std::invocable<decltype(cuda::get_property), upstream_with_both_properties, prop_with_value>, "");

__host__ __device__ constexpr bool test() {
  upstream_with_valueless_property with_valueless{};
  cuda::get_property(with_valueless, prop{});
  
  upstream_with_stateful_property with_value{};
  assert(cuda::get_property(with_value, prop_with_value{}) == 42);
  
  upstream_with_both_properties with_both{};
  cuda::get_property(with_both, prop{});
  assert(cuda::get_property(with_both, prop_with_value{}) == 42);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  return 0; 
}
