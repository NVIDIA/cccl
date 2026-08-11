// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/memory_pool>
#include <cuda/std/array>

#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using pool_ref_alias = cuda::device_memory_pool_ref;

[[nodiscard]] constexpr cuda::memory_pool_properties properties(const size_t release_threshold) noexcept
{
  cuda::memory_pool_properties props{};
  props.release_threshold = release_threshold;
  return props;
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

#if _CCCL_CTK_AT_LEAST(12, 9)
[[gnu::noinline]] void inspect_pinned(const cuda::pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_pinned_shared(const cuda::shared_pinned_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}
#endif // _CCCL_CTK_AT_LEAST(12, 9)

#if _CCCL_CTK_AT_LEAST(13, 0)
[[gnu::noinline]] void inspect_managed_no_init(const cuda::managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_managed_shared_no_init(const cuda::shared_managed_memory_pool& value)
{
  KEEP_FOR_DEBUGGER(value);
}
#endif // _CCCL_CTK_AT_LEAST(13, 0)

int main()
{
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

#if _CCCL_CTK_AT_LEAST(12, 9)
  const cuda::pinned_memory_pool pinned_pool{0, properties(8 * 1024 * 1024)};
  const cuda::shared_pinned_memory_pool shared_pinned_pool{0, properties(9 * 1024 * 1024)};
  inspect_pinned(pinned_pool);
  inspect_pinned_shared(shared_pinned_pool);
#endif // _CCCL_CTK_AT_LEAST(12, 9)

#if _CCCL_CTK_AT_LEAST(13, 0)
  // Managed pools are only inspected in their no_init state. Creating one needs
  // a 13.0+ driver and a device with concurrent managed access, and the harness
  // compares golden output exactly, so it has no way to skip a case whose
  // environment cannot supply it. A no_init pool never calls into the driver,
  // which keeps these two cases runnable everywhere the type exists.
  const cuda::managed_memory_pool managed_no_init_pool{cuda::no_init};
  const cuda::shared_managed_memory_pool shared_managed_no_init_pool{cuda::no_init};
  inspect_managed_no_init(managed_no_init_pool);
  inspect_managed_shared_no_init(shared_managed_no_init_pool);
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  return 0;
}
