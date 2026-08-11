// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Separate scenario so CMake can drop it whole on toolkits before CTK 13.0.
// Only no_init pools are inspected: creating a managed pool needs a 13.0+ driver
// and concurrent managed access, which the exact golden comparison cannot skip.

#include <cuda/memory_pool>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[gnu::noinline]] void inspect_no_init(const cuda::managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_ref(const cuda::managed_memory_pool_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_shared_no_init(const cuda::shared_managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  const cuda::managed_memory_pool no_init_pool{cuda::no_init};
  const cuda::shared_managed_memory_pool shared_no_init_pool{cuda::no_init};

  const cuda::managed_memory_pool_ref pool_reference{no_init_pool.get()};

  inspect_no_init(no_init_pool);
  inspect_ref(pool_reference);
  inspect_shared_no_init(shared_no_init_pool);

  return 0;
}
