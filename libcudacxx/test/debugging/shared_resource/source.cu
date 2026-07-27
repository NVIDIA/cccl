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

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_before_copy(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_after_copy(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_before_move(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

[[gnu::noinline]] void inspect_after_move(const shared_resource_type& resource)
{
  keep_for_debugger(resource);
}

int main()
{
  const shared_resource_type unique_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  const resource_alias aliased_resource = cuda::mr::make_shared_resource<managed_resource_type>();

  inspect_unique(unique_resource);
  inspect_alias(aliased_resource);

  // The copy and move scenarios stop twice on the same variable, so they show
  // whether a debugger refreshes what it reports for one object once its
  // ownership state changes.
  const shared_resource_type copy_source = cuda::mr::make_shared_resource<managed_resource_type>();
  inspect_before_copy(copy_source);
  const shared_resource_type copy_target = copy_source;
  inspect_after_copy(copy_source);

  // Moving exchanges the control-block pointer to null, so move_source renders
  // as empty afterwards. The header documents a moved-from shared_resource only
  // as "valid but unspecified", so this case pins the current implementation
  // rather than a documented guarantee.
  shared_resource_type move_source = cuda::mr::make_shared_resource<managed_resource_type>();
  inspect_before_move(move_source);
  const shared_resource_type move_target = cuda::std::move(move_source);
  inspect_after_move(move_source);
}
