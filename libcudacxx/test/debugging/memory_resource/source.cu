// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_resource>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

using device_resource_type      = cuda::mr::any_resource<cuda::mr::device_accessible>;
using host_device_resource_type = cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>;
using resource_alias            = device_resource_type;

[[gnu::noinline]] void inspect_device(const device_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_host_device(const host_device_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  keep_for_debugger(resource);
}

int main()
{
  using adapted_resource = cuda::mr::synchronous_resource_adapter<cuda::mr::legacy_managed_memory_resource>;
  const adapted_resource managed_resource{cuda::mr::legacy_managed_memory_resource{}};
  const device_resource_type device_resource{managed_resource};
  const host_device_resource_type host_device_resource{managed_resource};
  const resource_alias aliased_resource{managed_resource};

  inspect_device(device_resource);
  inspect_host_device(host_device_resource);
  inspect_alias(aliased_resource);
}
