// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(value) asm volatile("" : : "g"(&(value)) : "memory")

template <cuda::std::size_t Size>
struct test_resource
{
  cuda::std::byte storage[Size]{};

  void* allocate_sync(cuda::std::size_t, cuda::std::size_t)
  {
    return storage;
  }

  void deallocate_sync(void*, cuda::std::size_t, cuda::std::size_t) noexcept {}

  void* allocate(cuda::stream_ref, cuda::std::size_t, cuda::std::size_t)
  {
    return storage;
  }

  void deallocate(cuda::stream_ref, void*, cuda::std::size_t, cuda::std::size_t) noexcept {}

  friend bool operator==(const test_resource&, const test_resource&) noexcept
  {
    return true;
  }

  friend bool operator!=(const test_resource&, const test_resource&) noexcept
  {
    return false;
  }

  friend void get_property(const test_resource&, cuda::mr::device_accessible) noexcept {}
  friend void get_property(const test_resource&, cuda::mr::host_accessible) noexcept {}
};

using small_resource             = test_resource<1>;
using large_resource             = test_resource<64>;
using device_resource_type       = cuda::mr::any_resource<cuda::mr::device_accessible>;
using host_device_resource_type  = cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>;
using synchronous_resource_type  = cuda::mr::any_synchronous_resource<cuda::mr::device_accessible>;
using propertyless_resource_type = cuda::mr::any_resource<>;
using resource_alias             = device_resource_type;

[[gnu::noinline]] void inspect_in_situ(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_heap(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_synchronous(const synchronous_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_host_device(const host_device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_propertyless(const propertyless_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_empty(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_reset(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_reset(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_move(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_move_source(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_move_target(const device_resource_type& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

int main()
{
  const device_resource_type in_situ_resource{small_resource{}};
  const device_resource_type heap_resource{large_resource{}};
  const synchronous_resource_type synchronous_resource{small_resource{}};
  const host_device_resource_type host_device_resource{small_resource{}};
  const propertyless_resource_type propertyless_resource{small_resource{}};
  const resource_alias aliased_resource{small_resource{}};
  const device_resource_type empty_resource{};

  inspect_in_situ(in_situ_resource);
  inspect_heap(heap_resource);
  inspect_synchronous(synchronous_resource);
  inspect_host_device(host_device_resource);
  inspect_propertyless(propertyless_resource);
  inspect_alias(aliased_resource);
  inspect_empty(empty_resource);

  device_resource_type reset_resource{small_resource{}};
  inspect_before_reset(reset_resource);
  reset_resource.reset();
  inspect_after_reset(reset_resource);

  device_resource_type move_source{large_resource{}};
  inspect_before_move(move_source);
  device_resource_type move_target{cuda::std::move(move_source)};
  inspect_after_move_source(move_source);
  inspect_after_move_target(move_target);
}
