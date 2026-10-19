// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

template <cuda::std::size_t Size>
struct test_resource : cuda::mr::memory_resource_base<test_resource<Size>>
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

using resource_alias = cuda::mr::any_resource<cuda::mr::device_accessible>;

[[gnu::noinline]] void inspect_in_situ(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_heap(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void
inspect_synchronous(const cuda::mr::any_synchronous_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void
inspect_host_device(const cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_propertyless(const cuda::mr::any_resource<>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_alias(const resource_alias& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_empty(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_reset(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_reset(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_before_move(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_move_source(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

[[gnu::noinline]] void inspect_after_move_target(const cuda::mr::any_resource<cuda::mr::device_accessible>& resource)
{
  KEEP_FOR_DEBUGGER(resource);
}

int main()
{
  const cuda::mr::any_resource<cuda::mr::device_accessible> in_situ_resource{test_resource<1>{}};
  const cuda::mr::any_resource<cuda::mr::device_accessible> heap_resource{test_resource<64>{}};
  const cuda::mr::any_synchronous_resource<cuda::mr::device_accessible> synchronous_resource{test_resource<1>{}};
  const cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible> host_device_resource{
    test_resource<1>{}};
  const cuda::mr::any_resource<> propertyless_resource{test_resource<1>{}};
  const resource_alias aliased_resource{test_resource<1>{}};
  const cuda::mr::any_resource<cuda::mr::device_accessible> empty_resource{};

  inspect_in_situ(in_situ_resource);
  inspect_heap(heap_resource);
  inspect_synchronous(synchronous_resource);
  inspect_host_device(host_device_resource);
  inspect_propertyless(propertyless_resource);
  inspect_alias(aliased_resource);
  inspect_empty(empty_resource);

  cuda::mr::any_resource<cuda::mr::device_accessible> reset_resource{test_resource<1>{}};
  inspect_before_reset(reset_resource);
  reset_resource.reset();
  inspect_after_reset(reset_resource);

  cuda::mr::any_resource<cuda::mr::device_accessible> move_source{test_resource<64>{}};
  inspect_before_move(move_source);
  cuda::mr::any_resource<cuda::mr::device_accessible> move_target{cuda::std::move(move_source)};
  inspect_after_move_source(move_source);
  inspect_after_move_target(move_target);
}
