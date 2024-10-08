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

// cuda::mr::resource_with

#include <cuda/memory_resource>
#include <cuda/std/cstdint>

struct prop_with_value
{};
struct prop
{};

struct valid_resource_with_property
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const valid_resource_with_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_with_property&) const
  {
    return false;
  }
  friend void get_property(const valid_resource_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::resource_with<valid_resource_with_property, prop_with_value>, "");

struct valid_resource_without_property
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const valid_resource_without_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_without_property&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource_with<valid_resource_without_property, prop_with_value>, "");

struct invalid_resource_with_property
{
  friend void get_property(const invalid_resource_with_property&, prop_with_value) {}
};
static_assert(!cuda::mr::resource_with<invalid_resource_with_property, prop_with_value>, "");

struct resource_with_many_properties
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const resource_with_many_properties&) const
  {
    return true;
  }
  bool operator!=(const resource_with_many_properties&) const
  {
    return false;
  }
  friend void get_property(const resource_with_many_properties&, prop_with_value) {}
  friend void get_property(const resource_with_many_properties&, prop) {}
};
static_assert(cuda::mr::resource_with<resource_with_many_properties, prop_with_value, prop>, "");
static_assert(!cuda::mr::resource_with<resource_with_many_properties, prop_with_value, int, prop>, "");

struct derived_with_property : public valid_resource_without_property
{
  friend void get_property(const derived_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::resource_with<derived_with_property, prop_with_value>, "");

int main(int, char**)
{
  return 0;
}
