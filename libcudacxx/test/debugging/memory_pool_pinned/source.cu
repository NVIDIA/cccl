// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Separate scenario so CMake can drop it whole on toolkits before CTK 12.9.

#include <cuda/memory_pool>
#include <cuda/std/cstddef>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[nodiscard]] constexpr cuda::memory_pool_properties properties(const cuda::std::size_t release_threshold) noexcept
{
  cuda::memory_pool_properties props{};
  props.release_threshold = release_threshold;
  return props;
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

int main()
{
  const cuda::pinned_memory_pool owning_pool{0, properties(1024 * 1024)};
  const cuda::pinned_memory_pool referenced_pool{0, properties(3 * 1024 * 1024)};
  const cuda::shared_pinned_memory_pool shared_pool{0, properties(2 * 1024 * 1024)};

  const cuda::pinned_memory_pool_ref pool_reference{referenced_pool.get()};

  inspect_owning(owning_pool);
  inspect_ref(pool_reference);
  inspect_shared(shared_pool);

  return 0;
}
