//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

// Verify that the properties exist
static_assert(cuda::std::is_empty<cuda::mr::host_accessible>::value, "");
static_assert(cuda::std::is_empty<cuda::mr::device_accessible>::value, "");

// Verify that host accessible is the default if nothing is specified
static_assert(!cuda::mr::__is_host_accessible<>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_device_accessible<>, "");
static_assert(!cuda::mr::__is_device_accessible<cuda::mr::host_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that host device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_host_device_accessible<>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

struct host_only_resource
{
  friend constexpr void get_property(const host_only_resource&, cuda::mr::host_accessible) noexcept {}
};

struct device_only_resource
{
  friend constexpr void get_property(const device_only_resource&, cuda::mr::device_accessible) noexcept {}
};

struct host_device_resource
{
  friend constexpr void get_property(const host_device_resource&, cuda::mr::host_accessible) noexcept {}
  friend constexpr void get_property(const host_device_resource&, cuda::mr::device_accessible) noexcept {}
};

struct explicit_dynamic_resource
{
  friend constexpr void get_property(const explicit_dynamic_resource&, cuda::mr::host_accessible) noexcept {}
  friend constexpr cuda::mr::__memory_accessability
  get_property(const explicit_dynamic_resource&, cuda::mr::dynamic_accessibility_property) noexcept
  {
    return cuda::mr::__memory_accessability::__device;
  }
};

static_assert(cuda::has_property<host_only_resource, cuda::mr::dynamic_accessibility_property>);
static_assert(cuda::has_property<device_only_resource, cuda::mr::dynamic_accessibility_property>);
static_assert(cuda::has_property<host_device_resource, cuda::mr::dynamic_accessibility_property>);
static_assert(cuda::has_property<explicit_dynamic_resource, cuda::mr::dynamic_accessibility_property>);

static_assert(get_property(host_only_resource{}, cuda::mr::dynamic_accessibility_property{})
              == cuda::mr::__memory_accessability::__host);
static_assert(get_property(device_only_resource{}, cuda::mr::dynamic_accessibility_property{})
              == cuda::mr::__memory_accessability::__device);
static_assert(get_property(host_device_resource{}, cuda::mr::dynamic_accessibility_property{})
              == cuda::mr::__memory_accessability::__host_device);
static_assert(get_property(explicit_dynamic_resource{}, cuda::mr::dynamic_accessibility_property{})
              == cuda::mr::__memory_accessability::__device);

int main(int, char**)
{
  return 0;
}
