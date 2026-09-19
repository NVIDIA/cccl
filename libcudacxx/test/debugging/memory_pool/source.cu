// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_pool>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cstdio>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using pool_ref_alias = cuda::device_memory_pool_ref;

[[nodiscard]] constexpr cuda::memory_pool_properties properties(const cuda::std::size_t release_threshold) noexcept
{
  cuda::memory_pool_properties props{};
  props.release_threshold = release_threshold;
  return props;
}

// Stream-ordered allocation support is a device attribute, so probe it at runtime.
[[nodiscard]] bool device_pools_supported()
{
  try
  {
    (void) cuda::device_memory_pool{cuda::device_ref{0}};
    return true;
  }
  catch (const cuda::cuda_error&)
  {
    return false;
  }
}

[[gnu::noinline]] void inspect_owning(const cuda::device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_ref(const cuda::device_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_alias(const pool_ref_alias& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_no_init(const cuda::device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared(const cuda::shared_device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_copy(const cuda::shared_device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_no_init(const cuda::shared_device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::device_memory_pool_ref, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::device_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::device_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_in_use(const cuda::device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_release(const cuda::device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_in_use(const cuda::shared_device_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  if (!device_pools_supported())
  {
    std::puts("LIBCUDACXX_PRETTY_PRINTER_SKIP: device memory pools are not supported");
    return 0;
  }

  constexpr cuda::device_ref device{0};

  const cuda::device_memory_pool owning_pool{device, properties(1024 * 1024)};
  const cuda::device_memory_pool referenced_pool{device, properties(2 * 1024 * 1024)};
  const cuda::device_memory_pool aliased_pool{device, properties(3 * 1024 * 1024)};
  const cuda::device_memory_pool nested_pool{device, properties(4 * 1024 * 1024)};
  const cuda::device_memory_pool update_before_pool{device, properties(5 * 1024 * 1024)};
  const cuda::device_memory_pool update_after_pool{device, properties(6 * 1024 * 1024)};

  const cuda::device_memory_pool_ref pool_reference{referenced_pool.get()};
  const pool_ref_alias aliased_reference{aliased_pool.get()};
  const cuda::device_memory_pool no_init_pool{cuda::no_init};

  const cuda::shared_device_memory_pool shared_pool{device, properties(7 * 1024 * 1024)};
  const cuda::shared_device_memory_pool shared_no_init_pool{cuda::no_init};

  const cuda::std::array<cuda::device_memory_pool_ref, 2> nested_pools = {
    cuda::device_memory_pool_ref{nested_pool.get()}, cuda::device_memory_pool_ref{owning_pool.get()}};
  cuda::device_memory_pool_ref updated_reference{update_before_pool.get()};

  inspect_owning(owning_pool);
  inspect_ref(pool_reference);
  inspect_alias(aliased_reference);
  inspect_no_init(no_init_pool);
  inspect_shared(shared_pool);
  // Copying after the first inspection shows the reference count moving 1 -> 2.
  const cuda::shared_device_memory_pool shared_pool_copy{shared_pool};
  inspect_shared_copy(shared_pool_copy);
  inspect_shared_no_init(shared_no_init_pool);
  inspect_nested(nested_pools);
  inspect_before_update(updated_reference);
  updated_reference = cuda::device_memory_pool_ref{update_after_pool.get()};
  inspect_after_update(updated_reference);

  constexpr cuda::std::size_t allocation_size = 1024 * 1024;
  cuda::stream stream{device};

  cuda::device_memory_pool in_use_pool{device, properties(8 * 1024 * 1024)};
  void* const in_use_allocation = in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_in_use(in_use_pool);

  // A zero release threshold returns the memory to the driver on synchronization.
  cuda::device_memory_pool released_pool{device, properties(0)};
  void* const released_allocation = released_pool.allocate(stream, allocation_size);
  released_pool.deallocate(stream, released_allocation, allocation_size);
  stream.sync();
  inspect_after_release(released_pool);

  in_use_pool.deallocate(stream, in_use_allocation, allocation_size);
  stream.sync();

  cuda::shared_device_memory_pool shared_in_use_pool{device, properties(9 * 1024 * 1024)};
  void* const shared_in_use_allocation = shared_in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_shared_in_use(shared_in_use_pool);

  shared_in_use_pool.deallocate(stream, shared_in_use_allocation, allocation_size);
  stream.sync();

  return 0;
}
