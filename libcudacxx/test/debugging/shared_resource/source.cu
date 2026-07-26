// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_resource>
#include <cuda/std/utility>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

using managed_resource_type = cuda::mr::legacy_managed_memory_resource;
using shared_resource_type  = cuda::mr::shared_resource<managed_resource_type>;
using resource_alias        = shared_resource_type;

[[gnu::noinline]] void inspect_unique(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_shared(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_moved_into(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_moved_from(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

int main()
{
  const shared_resource_type unique_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  const shared_resource_type first_reference  = cuda::mr::make_shared_resource<managed_resource_type>();
  const shared_resource_type second_reference = first_reference;

  const resource_alias aliased_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  // Moving exchanges the control-block pointer to null, so donor_resource
  // renders as empty below. The header documents a moved-from shared_resource
  // only as "valid but unspecified", so this case pins the current
  // implementation rather than a documented guarantee.
  shared_resource_type donor_resource       = cuda::mr::make_shared_resource<managed_resource_type>();
  const shared_resource_type moved_resource = cuda::std::move(donor_resource);

  inspect_unique(unique_resource);
  inspect_shared(second_reference);
  inspect_alias(aliased_resource);
  inspect_moved_into(moved_resource);
  inspect_moved_from(donor_resource);
}
