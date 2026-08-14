// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Separate scenario so CMake can drop it whole on toolkits before CTK 13.0.

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

// Creating a managed pool also needs a 13.0 driver, which the toolkit gate in CMake
// cannot see, so probe it here and let the harness skip the scenario without one.
// Returns the driver's error message, or an empty string when managed pools are usable.
[[nodiscard]] std::string managed_pools_error()
{
  try
  {
    (void) cuda::managed_memory_pool{};
    return {};
  }
  catch (const cuda::cuda_error& err)
  {
    return err.what();
  }
}

[[gnu::noinline]] void inspect_owning(const cuda::managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_ref(const cuda::managed_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_no_init(const cuda::managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared(const cuda::shared_managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_no_init(const cuda::shared_managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_in_use(const cuda::managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_in_use(const cuda::shared_managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  if (const std::string error = managed_pools_error(); !error.empty())
  {
    std::printf("LIBCUDACXX_PRETTY_PRINTER_SKIP: managed memory pools are not supported: %s\n", error.c_str());
    return 0;
  }

  const cuda::managed_memory_pool owning_pool{properties(1024 * 1024)};
  const cuda::managed_memory_pool referenced_pool{properties(3 * 1024 * 1024)};
  const cuda::managed_memory_pool no_init_pool{cuda::no_init};

  const cuda::managed_memory_pool_ref pool_reference{referenced_pool.get()};

  const cuda::shared_managed_memory_pool shared_pool{properties(2 * 1024 * 1024)};
  const cuda::shared_managed_memory_pool shared_no_init_pool{cuda::no_init};

  inspect_owning(owning_pool);
  inspect_ref(pool_reference);
  inspect_no_init(no_init_pool);
  inspect_shared(shared_pool);
  inspect_shared_no_init(shared_no_init_pool);

  constexpr cuda::std::size_t allocation_size = 1024 * 1024;
  cuda::stream stream{cuda::device_ref{0}};

  cuda::managed_memory_pool in_use_pool{properties(4 * 1024 * 1024)};
  void* const in_use_allocation = in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_in_use(in_use_pool);

  cuda::shared_managed_memory_pool shared_in_use_pool{properties(5 * 1024 * 1024)};
  void* const shared_in_use_allocation = shared_in_use_pool.allocate(stream, allocation_size);
  stream.sync();
  inspect_shared_in_use(shared_in_use_pool);

  in_use_pool.deallocate(stream, in_use_allocation, allocation_size);
  shared_in_use_pool.deallocate(stream, shared_in_use_allocation, allocation_size);
  stream.sync();

  return 0;
}
