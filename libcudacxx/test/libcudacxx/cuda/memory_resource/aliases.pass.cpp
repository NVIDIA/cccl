//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// UNSUPPORTED: nvrtc

// cuda::mr convenience aliases for resource_ref and any_resource

#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

template <class... Properties>
struct test_resource
{
  void* allocate_sync(cuda::std::size_t, cuda::std::size_t)
  {
    return nullptr;
  }

  void deallocate_sync(void*, cuda::std::size_t, cuda::std::size_t) noexcept {}

  void* allocate(cuda::stream_ref, cuda::std::size_t, cuda::std::size_t)
  {
    return nullptr;
  }

  void deallocate(cuda::stream_ref, void*, cuda::std::size_t, cuda::std::size_t) noexcept {}

  bool operator==(const test_resource&) const
  {
    return true;
  }
  bool operator!=(const test_resource&) const
  {
    return false;
  }

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES(::cuda::std::__is_included_in_v<Property, Properties...>)
  friend void get_property(const test_resource&, Property) noexcept {}
};

using device_res      = test_resource<cuda::mr::device_accessible>;
using host_res        = test_resource<cuda::mr::host_accessible>;
using host_device_res = test_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>;

using cuda::std::is_constructible_v;
using cuda::std::is_same_v;

static_assert(is_same_v<cuda::mr::device_resource_ref, cuda::mr::resource_ref<cuda::mr::device_accessible>>);
static_assert(is_same_v<cuda::mr::host_resource_ref, cuda::mr::resource_ref<cuda::mr::host_accessible>>);
static_assert(is_same_v<cuda::mr::host_device_resource_ref,
                        cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>>);
static_assert(is_same_v<cuda::mr::any_device_resource, cuda::mr::any_resource<cuda::mr::device_accessible>>);
static_assert(is_same_v<cuda::mr::any_host_resource, cuda::mr::any_resource<cuda::mr::host_accessible>>);
static_assert(is_same_v<cuda::mr::any_host_device_resource,
                        cuda::mr::any_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>>);

static_assert(is_constructible_v<cuda::mr::device_resource_ref, device_res&>);
static_assert(!is_constructible_v<cuda::mr::device_resource_ref, host_res&>);
static_assert(is_constructible_v<cuda::mr::host_resource_ref, host_res&>);
static_assert(!is_constructible_v<cuda::mr::host_resource_ref, device_res&>);
static_assert(is_constructible_v<cuda::mr::host_device_resource_ref, host_device_res&>);
static_assert(!is_constructible_v<cuda::mr::host_device_resource_ref, device_res&>);
static_assert(!is_constructible_v<cuda::mr::host_device_resource_ref, host_res&>);

static_assert(is_constructible_v<cuda::mr::any_device_resource, device_res>);
static_assert(!is_constructible_v<cuda::mr::any_device_resource, host_res>);
static_assert(is_constructible_v<cuda::mr::any_host_resource, host_res>);
static_assert(!is_constructible_v<cuda::mr::any_host_resource, device_res>);
static_assert(is_constructible_v<cuda::mr::any_host_device_resource, host_device_res>);
static_assert(!is_constructible_v<cuda::mr::any_host_device_resource, device_res>);
static_assert(!is_constructible_v<cuda::mr::any_host_device_resource, host_res>);

static_assert(is_constructible_v<cuda::mr::device_resource_ref, cuda::mr::host_device_resource_ref>);
static_assert(is_constructible_v<cuda::mr::host_resource_ref, cuda::mr::host_device_resource_ref>);
static_assert(is_constructible_v<cuda::mr::any_device_resource, cuda::mr::any_host_device_resource>);
static_assert(is_constructible_v<cuda::mr::any_host_resource, cuda::mr::any_host_device_resource>);

int main(int, char**)
{
  return 0;
}
