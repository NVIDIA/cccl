// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Separate scenario so CMake can drop it whole on toolkits before CTK 12.9.

#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cstdio>
#include <string>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[nodiscard]] constexpr cuda::memory_pool_properties properties(const cuda::std::size_t release_threshold) noexcept
{
  cuda::memory_pool_properties props{};
  props.release_threshold = release_threshold;
  return props;
}

// Host pool support is a device attribute the toolkit gate in CMake cannot see.
// Returns the driver's error message, or an empty string when pinned pools are usable.
[[nodiscard]] std::string pinned_pools_error()
{
  try
  {
    (void) cuda::pinned_memory_pool{0};
    return {};
  }
  catch (const cuda::cuda_error& err)
  {
    return err.what();
  }
}

[[gnu::noinline]] void inspect_owning(const cuda::pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_ref(const cuda::pinned_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared(const cuda::shared_pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_in_use(const cuda::pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_in_use(const cuda::shared_pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  if (const std::string error = pinned_pools_error(); !error.empty())
  {
    std::printf("LIBCUDACXX_PRETTY_PRINTER_SKIP: pinned memory pools are not supported: %s\n", error.c_str());
    return 0;
  }

  const cuda::pinned_memory_pool owning_pool{0, properties(1024 * 1024)};
  const cuda::pinned_memory_pool referenced_pool{0, properties(3 * 1024 * 1024)};
  const cuda::shared_pinned_memory_pool shared_pool{0, properties(2 * 1024 * 1024)};

  const cuda::pinned_memory_pool_ref pool_reference{referenced_pool.get()};

  inspect_owning(owning_pool);
  inspect_ref(pool_reference);
  inspect_shared(shared_pool);

  constexpr cuda::std::size_t allocation_size = 1024 * 1024;
  cuda::stream stream{cuda::device_ref{0}};

  cuda::pinned_memory_pool in_use_pool{0, properties(4 * 1024 * 1024)};
  void* const in_use_allocation = in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_in_use(in_use_pool);

  cuda::shared_pinned_memory_pool shared_in_use_pool{0, properties(5 * 1024 * 1024)};
  void* const shared_in_use_allocation = shared_in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_shared_in_use(shared_in_use_pool);

  in_use_pool.deallocate(stream, in_use_allocation, allocation_size);
  shared_in_use_pool.deallocate(stream, shared_in_use_allocation, allocation_size);
  stream.sync();

  return 0;
}
