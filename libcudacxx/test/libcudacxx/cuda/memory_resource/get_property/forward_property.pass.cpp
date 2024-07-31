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
// cuda::forward_property

#include <cuda/memory_resource>
#include <cuda/std/cassert>

namespace has_upstream_resource
{
struct Upstream
{};

__device__ Upstream upstream{};

struct with_reference
{
  Upstream& upstream_resource() const
  {
    return upstream;
  }
};
static_assert(cuda::__has_upstream_resource<with_reference, Upstream>, "");

struct with_const_reference
{
  const Upstream& upstream_resource() const
  {
    return upstream;
  }
};
static_assert(cuda::__has_upstream_resource<with_const_reference, Upstream>, "");

struct with_value
{
  Upstream upstream_resource() const
  {
    return Upstream{};
  }
};
static_assert(cuda::__has_upstream_resource<with_value, Upstream>, "");

struct with_const_value
{
  const Upstream upstream_resource() const
  {
    return Upstream{};
  }
};
static_assert(cuda::__has_upstream_resource<with_const_value, Upstream>, "");

struct Convertible
{
  operator Upstream()
  {
    return Upstream{};
  }
};

struct with_conversion
{
  Convertible upstream_resource() const
  {
    return Convertible{};
  }
};
static_assert(!cuda::__has_upstream_resource<with_conversion, Upstream>, "");
} // namespace has_upstream_resource

namespace forward_property
{
struct prop_with_value
{
  using value_type = int;
};
struct prop
{};

template <class Upstream>
struct derived_plain : public cuda::forward_property<derived_plain<Upstream>, Upstream>
{
  constexpr Upstream upstream_resource() const noexcept
  {
    return Upstream{};
  }
};

struct upstream_with_valueless_property
{
  friend constexpr void get_property(const upstream_with_valueless_property&, prop) {}
};
static_assert(cuda::has_property<derived_plain<upstream_with_valueless_property>, prop>, "");
static_assert(!cuda::has_property<derived_plain<upstream_with_valueless_property>, prop_with_value>, "");

struct upstream_with_stateful_property
{
  friend constexpr int get_property(const upstream_with_stateful_property&, prop_with_value)
  {
    return 42;
  }
};
static_assert(!cuda::has_property<derived_plain<upstream_with_stateful_property>, prop>, "");
static_assert(cuda::has_property<derived_plain<upstream_with_stateful_property>, prop_with_value>, "");

struct upstream_with_both_properties
{
  friend constexpr void get_property(const upstream_with_both_properties&, prop) {}
  friend constexpr int get_property(const upstream_with_both_properties&, prop_with_value)
  {
    return 42;
  }
};
static_assert(cuda::has_property<derived_plain<upstream_with_both_properties>, prop>, "");
static_assert(cuda::has_property<derived_plain<upstream_with_both_properties>, prop_with_value>, "");

struct derived_override : public cuda::forward_property<derived_override, upstream_with_both_properties>
{
  constexpr upstream_with_both_properties upstream_resource() const noexcept
  {
    return upstream_with_both_properties{};
  }
  // Get called directly so needs to be annotated
  __host__ __device__ friend constexpr int get_property(const derived_override&, prop_with_value)
  {
    return 1337;
  }
};

struct convertible_to_upstream
{
  operator upstream_with_both_properties() const noexcept
  {
    return upstream_with_both_properties{};
  }
};

struct derived_with_converstin_upstream_resource
    : public cuda::forward_property<derived_with_converstin_upstream_resource, upstream_with_both_properties>
{
  constexpr convertible_to_upstream upstream_resource() const noexcept
  {
    return convertible_to_upstream{};
  }
};
static_assert(!cuda::has_property<derived_with_converstin_upstream_resource, prop_with_value>, "");

__host__ __device__ constexpr bool test_stateful()
{
  using derived_no_override = derived_plain<upstream_with_stateful_property>;
  const derived_no_override without_override{};
  assert(get_property(without_override, prop_with_value{}) == 42);

  const derived_override with_override{};
  assert(get_property(with_override, prop_with_value{}) == 1337);

  return true;
}
} // namespace forward_property

int main(int, char**)
{
  forward_property::test_stateful();
  static_assert(forward_property::test_stateful(), "");
  return 0;
}
