//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__device/all_devices.h>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <testing.cuh>

#if _CCCL_CTK_AT_LEAST(13, 0) && !_CCCL_OS(WINDOWS)
#  define SHARED_TEST_TYPES \
    cuda::shared_managed_memory_pool, cuda::shared_device_memory_pool, cuda::shared_pinned_memory_pool
#elif _CCCL_CTK_AT_LEAST(12, 9)
#  define SHARED_TEST_TYPES cuda::shared_device_memory_pool, cuda::shared_pinned_memory_pool
#else
#  define SHARED_TEST_TYPES cuda::shared_device_memory_pool
#endif

template <typename PoolType>
PoolType construct_shared_pool(cuda::memory_pool_properties props = {})
{
  if constexpr (cuda::std::is_same_v<PoolType, cuda::shared_device_memory_pool>)
  {
    return cuda::shared_device_memory_pool(0, props);
  }
#if _CCCL_CTK_AT_LEAST(12, 9)
  else if constexpr (cuda::std::is_same_v<PoolType, cuda::shared_pinned_memory_pool>)
  {
    return cuda::shared_pinned_memory_pool(0, props);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 9)
#if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (cuda::std::is_same_v<PoolType, cuda::shared_managed_memory_pool>)
  {
    return cuda::shared_managed_memory_pool(props);
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)
}

// --- static assertions ---

template <typename PoolType>
void shared_pool_static_asserts()
{
  // Shared pools must be copyable (unlike owning pools)
  static_assert(cuda::std::is_copy_constructible<PoolType>::value);
  static_assert(cuda::std::is_copy_assignable<PoolType>::value);
  static_assert(cuda::std::is_move_constructible<PoolType>::value);
  static_assert(cuda::std::is_move_assignable<PoolType>::value);
  static_assert(!cuda::std::is_trivially_destructible<PoolType>::value);
}

template void shared_pool_static_asserts<cuda::shared_device_memory_pool>();
#if _CCCL_CTK_AT_LEAST(13, 0)
template void shared_pool_static_asserts<cuda::shared_managed_memory_pool>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)
#if _CCCL_CTK_AT_LEAST(12, 9)
template void shared_pool_static_asserts<cuda::shared_pinned_memory_pool>();
#endif // _CCCL_CTK_AT_LEAST(12, 9)

// --- construction ---

C2H_CCCLRT_TEST_LIST("shared_memory_pool construction", "[memory_resource]", SHARED_TEST_TYPES)
{
  using shared_pool = TestType;

  SECTION("Construct and get handle")
  {
    shared_pool pool = construct_shared_pool<shared_pool>();
    CHECK(pool.get() != nullptr);
  }

  SECTION("Construct with no_init")
  {
    shared_pool pool(cuda::no_init);
    CHECK(pool.get() == nullptr);
  }

  SECTION("from_native_handle")
  {
    // Create an owning pool, release the handle, and wrap it via from_native_handle.
    cuda::device_memory_pool owning_pool{cuda::device_ref{0}};
    cudaMemPool_t raw    = owning_pool.release();
    shared_pool from_raw = shared_pool::from_native_handle(raw);
    CHECK(from_raw.get() == raw);
    // from_raw owns the handle and will destroy it on scope exit.
  }
}

// --- copy / move semantics ---

C2H_CCCLRT_TEST_LIST("shared_memory_pool copy and move", "[memory_resource]", SHARED_TEST_TYPES)
{
  using shared_pool = TestType;

  shared_pool pool = construct_shared_pool<shared_pool>();
  auto handle      = pool.get();

  SECTION("Copy construction shares the pool")
  {
    shared_pool copy(pool); // NOLINT(performance-unnecessary-copy-initialization)
    CHECK(copy.get() == handle);
    CHECK(pool.get() == handle);
  }

  SECTION("Copy assignment shares the pool")
  {
    shared_pool copy(cuda::no_init);
    copy = pool;
    CHECK(copy.get() == handle);
    CHECK(pool.get() == handle);
  }

  SECTION("Move construction transfers ownership")
  {
    shared_pool moved(cuda::std::move(pool));
    CHECK(moved.get() == handle);
  }

  SECTION("Move assignment transfers ownership")
  {
    shared_pool moved(cuda::no_init);
    moved = cuda::std::move(pool);
    CHECK(moved.get() == handle);
  }

  SECTION("Multiple copies keep pool alive")
  {
    shared_pool outer = construct_shared_pool<shared_pool>();
    auto saved_handle = outer.get();
    {
      shared_pool copy1(outer); // NOLINT(performance-unnecessary-copy-initialization)
      shared_pool copy2(outer); // NOLINT(performance-unnecessary-copy-initialization)
      CHECK(copy1.get() == saved_handle);
      CHECK(copy2.get() == saved_handle);
      // copy1, copy2 destroyed here — pool should survive
    }
    // Pool still works after copies are gone
    CHECK(outer.get() == saved_handle);
    void* ptr = outer.allocate_sync(64);
    CHECK(ptr != nullptr);
    outer.deallocate_sync(ptr, 64);
  }
}

// --- equality ---

C2H_CCCLRT_TEST_LIST("shared_memory_pool comparison", "[memory_resource]", SHARED_TEST_TYPES)
{
  using shared_pool = TestType;

  shared_pool pool1 = construct_shared_pool<shared_pool>();
  shared_pool pool2 = construct_shared_pool<shared_pool>();

  SECTION("Different pools are not equal")
  {
    CHECK(pool1 != pool2);
  }

  SECTION("Copies are equal")
  {
    shared_pool copy(pool1); // NOLINT(performance-unnecessary-copy-initialization)
    CHECK(pool1 == copy);
  }

  SECTION("Self equality")
  {
    CHECK(pool1 == pool1);
  }
}

// --- pool operations ---

C2H_CCCLRT_TEST_LIST("shared_memory_pool operations", "[memory_resource]", SHARED_TEST_TYPES)
{
  using shared_pool = TestType;

  shared_pool pool = construct_shared_pool<shared_pool>();

  SECTION("allocate and deallocate")
  {
    cuda::stream stream{cuda::device_ref{0}};
    void* ptr = pool.allocate(stream, 1024, cuda::mr::default_cuda_malloc_alignment);
    CHECK(ptr != nullptr);
    pool.deallocate(stream, ptr, 1024, cuda::mr::default_cuda_malloc_alignment);
    stream.sync();
  }

  SECTION("allocate_sync and deallocate_sync")
  {
    void* ptr = pool.allocate_sync(1024);
    CHECK(ptr != nullptr);
    pool.deallocate_sync(ptr, 1024);
  }

  SECTION("trim_to")
  {
    cuda::stream stream{cuda::device_ref{0}};
    void* ptr = pool.allocate(stream, 2048 * sizeof(int));
    pool.deallocate(stream, ptr, 2048 * sizeof(int));
    stream.sync();
    pool.trim_to(0);
  }

  SECTION("attribute access")
  {
    size_t threshold = pool.attribute(cuda::memory_pool_attributes::release_threshold);
    CHECK(threshold == cuda::std::numeric_limits<size_t>::max());
  }

  SECTION("set_attribute")
  {
    bool attr = pool.attribute(cuda::memory_pool_attributes::reuse_follow_event_dependencies);
    pool.set_attribute(cuda::memory_pool_attributes::reuse_follow_event_dependencies, !attr);
    bool new_attr = pool.attribute(cuda::memory_pool_attributes::reuse_follow_event_dependencies);
    CHECK(attr == !new_attr);
  }

  SECTION("Operations work through a copy")
  {
    shared_pool copy(pool);
    cuda::stream stream{cuda::device_ref{0}};
    void* ptr = copy.allocate(stream, 512);
    CHECK(ptr != nullptr);
    copy.deallocate(stream, ptr, 512);
    stream.sync();
  }
}

// --- resource concept ---

C2H_CCCLRT_TEST("shared_device_memory_pool satisfies resource_with", "[memory_resource]")
{
  static_assert(cuda::mr::resource_with<cuda::shared_device_memory_pool, cuda::mr::device_accessible>);

  cuda::shared_device_memory_pool pool{cuda::device_ref{0}};
  cuda::mr::resource_ref<cuda::mr::device_accessible> ref = pool;
  (void) ref;
}

#if _CCCL_CTK_AT_LEAST(12, 9)
C2H_CCCLRT_TEST("shared_pinned_memory_pool satisfies resource_with", "[memory_resource]")
{
  static_assert(cuda::mr::resource_with<cuda::shared_pinned_memory_pool, cuda::mr::device_accessible>);
  static_assert(cuda::mr::resource_with<cuda::shared_pinned_memory_pool, cuda::mr::host_accessible>);

  cuda::shared_pinned_memory_pool pool{0};
  cuda::mr::resource_ref<cuda::mr::device_accessible, cuda::mr::host_accessible> ref = pool;
  (void) ref;
}
#endif // _CCCL_CTK_AT_LEAST(12, 9)

#if _CCCL_CTK_AT_LEAST(13, 0) && !_CCCL_OS(WINDOWS)
C2H_CCCLRT_TEST("shared_managed_memory_pool satisfies resource_with", "[memory_resource]")
{
  static_assert(cuda::mr::resource_with<cuda::shared_managed_memory_pool, cuda::mr::device_accessible>);
  static_assert(cuda::mr::resource_with<cuda::shared_managed_memory_pool, cuda::mr::host_accessible>);

  cuda::shared_managed_memory_pool pool{};
  cuda::mr::resource_ref<cuda::mr::device_accessible, cuda::mr::host_accessible> ref = pool;
  (void) ref;
}
#endif // _CCCL_CTK_AT_LEAST(13, 0) && !_CCCL_OS(WINDOWS)
