// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_resource>
#include <cuda/std/utility>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using managed_resource_type = cuda::mr::legacy_managed_memory_resource;
using shared_resource_type  = cuda::mr::shared_resource<managed_resource_type>;
using resource_alias        = shared_resource_type;

[[gnu::noinline]] void inspect_unique(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_copy(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_copy_source(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_copy_target(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_move(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_move_source(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_move_target(const shared_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

int main()
{
  const shared_resource_type unique_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  const resource_alias aliased_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  inspect_unique(unique_resource);
  inspect_alias(aliased_resource);

  // The copy scenario stops before and after the copy, so it shows whether a
  // debugger refreshes the use count it reports for one handle once a second
  // handle starts sharing the same resource.
  const shared_resource_type copy_source = cuda::mr::make_shared_resource<managed_resource_type>();
  inspect_before_copy(copy_source);
  const shared_resource_type copy_target = copy_source;
  inspect_copy_source(copy_source);
  inspect_copy_target(copy_target);

  // Moving exchanges the control-block pointer to null, so move_source renders
  // as empty afterwards. The header documents a moved-from shared_resource only
  // as "valid but unspecified", so this case pins the current implementation
  // rather than a documented guarantee.
  shared_resource_type move_source = cuda::mr::make_shared_resource<managed_resource_type>();
  inspect_before_move(move_source);
  const shared_resource_type move_target = cuda::std::move(move_source);
  inspect_move_source(move_source);
  inspect_move_target(move_target);
}
