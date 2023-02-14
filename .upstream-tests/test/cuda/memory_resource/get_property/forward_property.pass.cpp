//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// cuda::forward_property
#include <cuda/std/cassert>
#include <cuda/memory_resource>

struct prop_with_value {
  using value_type = int;
};
struct prop {};

template<class Upstream>
struct derived_plain : public cuda::forward_property<derived_plain<Upstream>, Upstream> 
{
  __host__ __device__  constexpr Upstream upstream_resource() const noexcept { return Upstream{}; }
};

struct upstream_with_valueless_property {
  __host__ __device__ friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};
static_assert( cuda::has_property<derived_plain<upstream_with_valueless_property>, prop>, "");
static_assert(!cuda::has_property<derived_plain<upstream_with_valueless_property>, prop_with_value>, "");

struct upstream_with_stateful_property {
  __host__ __device__ friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value) {
    return 42;
  }
};
static_assert(!cuda::has_property<derived_plain<upstream_with_stateful_property>, prop>, "");
static_assert( cuda::has_property<derived_plain<upstream_with_stateful_property>, prop_with_value>, "");

struct upstream_with_both_properties {
  __host__ __device__ friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  __host__ __device__ friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value) {
    return 42;
  }
};
static_assert( cuda::has_property<derived_plain<upstream_with_both_properties>, prop>, "");
static_assert( cuda::has_property<derived_plain<upstream_with_both_properties>, prop_with_value>, "");

struct derived_override : public cuda::forward_property<derived_override, upstream_with_both_properties> 
{
  __host__ __device__  constexpr upstream_with_both_properties upstream_resource() const noexcept { 
    return upstream_with_both_properties{}; 
  }
  __host__ __device__ friend constexpr int get_property(const derived_override&, prop_with_value) {
    return 1337;
  }
};

__host__ __device__ constexpr bool test_stateful() {
  using derived_no_override = derived_plain<upstream_with_stateful_property>;
  const derived_no_override without_override{};
  assert(get_property(without_override, prop_with_value{}) == 42);
  
  const derived_override with_override{};
  assert(get_property(with_override, prop_with_value{}) == 1337);

  return true;
}

int main(int, char**) {
  test_stateful();
  static_assert(test_stateful(), "");
  return 0; 
}
