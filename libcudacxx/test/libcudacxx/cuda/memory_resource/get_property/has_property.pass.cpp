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

// cuda::has_property, cuda::has_property_with

#include <cuda/memory_resource>

struct prop_with_value
{
  using value_type = int;
};
struct prop
{};

static_assert(cuda::property_with_value<prop_with_value>, "");
static_assert(!cuda::property_with_value<prop>, "");

struct valid_property
{
  friend void get_property(const valid_property&, prop) {}
};
static_assert(!cuda::has_property<valid_property, prop_with_value>, "");
static_assert(cuda::has_property<valid_property, prop>, "");
static_assert(!cuda::has_property_with<valid_property, prop, int>, "");

struct valid_property_with_value
{
  friend int get_property(const valid_property_with_value&, prop_with_value)
  {
    return 42;
  }
};
static_assert(cuda::has_property<valid_property_with_value, prop_with_value>, "");
static_assert(!cuda::has_property<valid_property_with_value, prop>, "");
static_assert(cuda::has_property_with<valid_property_with_value, prop_with_value, int>, "");
static_assert(!cuda::has_property_with<valid_property_with_value, prop_with_value, double>, "");

struct derived_from_property : public valid_property
{
  friend int get_property(const derived_from_property&, prop_with_value)
  {
    return 42;
  }
};
static_assert(cuda::has_property<derived_from_property, prop_with_value>, "");
static_assert(cuda::has_property<derived_from_property, prop>, "");
static_assert(cuda::has_property_with<derived_from_property, prop_with_value, int>, "");
static_assert(!cuda::has_property_with<derived_from_property, prop_with_value, double>, "");

int main(int, char**)
{
  return 0;
}
