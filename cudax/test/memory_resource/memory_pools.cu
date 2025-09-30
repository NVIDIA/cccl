//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <stdexcept>

#include <testing.cuh>

#if _CCCL_CUDACC_AT_LEAST(13, 0)
#  define TEST_TYPES cudax::managed_memory_pool, cudax::device_memory_pool, cudax::pinned_memory_pool
#elif _CCCL_CUDACC_AT_LEAST(12, 6)
#  define TEST_TYPES cudax::device_memory_pool, cudax::pinned_memory_pool
#else
#  define TEST_TYPES cudax::device_memory_pool
#endif

namespace cudax = cuda::experimental;

template <typename PoolType>
void pool_static_asserts()
{
  static_assert(!cuda::std::is_trivial<PoolType>::value, "");
  static_assert(!cuda::std::is_trivially_default_constructible<PoolType>::value, "");
  static_assert(!cuda::std::is_copy_constructible<PoolType>::value, "");
  static_assert(!cuda::std::is_move_constructible<PoolType>::value, "");
  static_assert(!cuda::std::is_copy_assignable<PoolType>::value, "");
  static_assert(!cuda::std::is_move_assignable<PoolType>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<PoolType>::value, "");
  static_assert(!cuda::std::is_empty<PoolType>::value, "");
}

#if _CCCL_CUDACC_AT_LEAST(13, 0)
template void pool_static_asserts<cudax::managed_memory_pool>();
#endif
#if _CCCL_CUDACC_AT_LEAST(12, 6)
template void pool_static_asserts<cudax::pinned_memory_pool>();
#endif
template void pool_static_asserts<cudax::device_memory_pool>();

#if _CCCL_CUDACC_AT_LEAST(13, 0)
static_assert(cuda::std::is_default_constructible<cudax::managed_memory_pool>::value, "");
static_assert(cuda::std::is_default_constructible<cudax::pinned_memory_pool>::value, "");
#endif
static_assert(!cuda::std::is_default_constructible<cudax::device_memory_pool>::value, "");

template <typename PoolType>
PoolType construct_pool([[maybe_unused]] int device_id, cudax::memory_pool_properties props = {})
{
  if constexpr (cuda::std::is_same_v<PoolType, cudax::device_memory_pool>)
  {
    return cudax::device_memory_pool(device_id, props);
  }
  else
  {
#if _CCCL_CTK_AT_LEAST(12, 6)
    if constexpr (cuda::std::is_same_v<PoolType, cudax::pinned_memory_pool>)
    {
      return cudax::pinned_memory_pool(0, props);
    }
    else
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return cudax::managed_memory_pool(props);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
    }
#endif // _CCCL_CTK_AT_LEAST(12, 6)
  }
}

static bool ensure_release_threshold(::cudaMemPool_t pool, const size_t expected_threshold)
{
  size_t release_threshold = expected_threshold + 1337; // use something different than the expected threshold
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolGetAttribute,
    "Failed to call cudaMemPoolGetAttribute",
    pool,
    ::cudaMemPoolAttrReleaseThreshold,
    &release_threshold);
  return release_threshold == expected_threshold;
}

static bool ensure_disable_reuse(::cudaMemPool_t pool)
{
  int disable_reuse = 0;
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolGetAttribute,
    "Failed to call cudaMemPoolGetAttribute",
    pool,
    ::cudaMemPoolReuseAllowOpportunistic,
    &disable_reuse);

  return disable_reuse != 0;
}

static bool ensure_export_handle(::cudaMemPool_t pool, const ::cudaMemAllocationHandleType allocation_handle)
{
  size_t handle              = 0;
  const ::cudaError_t status = ::cudaMemPoolExportToShareableHandle(&handle, pool, allocation_handle, 0);
  ::cudaGetLastError(); // Clear CUDA error state

  // If no export was defined we need to query cudaErrorInvalidValue
  return allocation_handle == ::cudaMemHandleTypeNone ? status == ::cudaErrorInvalidValue : status == ::cudaSuccess;
}

C2H_CCCLRT_TEST_LIST("device_memory_pool construction", "[memory_resource]", TEST_TYPES)
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with with cudaGetDevice.", &current_device);
  }

  int driver_version = 0;
  {
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &driver_version);
  }

  ::cudaMemPool_t current_default_pool{};
  {
    _CCCL_TRY_CUDA_API(::cudaDeviceGetDefaultMemPool,
                       "Failed to call cudaDeviceGetDefaultMemPool",
                       &current_default_pool,
                       current_device);
  }

  using memory_pool = TestType;
  SECTION("Construct from device id")
  {
    memory_pool from_device = construct_pool<memory_pool>(current_device);

    ::cudaMemPool_t get = from_device.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, cuda::std::numeric_limits<size_t>::max()));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with empty properties")
  {
    cudax::memory_pool_properties props{};
    memory_pool from_defaulted_properties = construct_pool<memory_pool>(current_device, props);

    ::cudaMemPool_t get = from_defaulted_properties.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, cuda::std::numeric_limits<size_t>::max()));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with initial pool size")
  {
    cudax::memory_pool_properties props = {20, 42};
    memory_pool with_threshold          = construct_pool<memory_pool>(current_device, props);

    ::cudaMemPool_t get = with_threshold.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, props.release_threshold));

    // Ensure that we disable reuse
    CHECK(ensure_disable_reuse(get));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  if (cuda::std::is_same_v<memory_pool, cudax::device_memory_pool>)
  {
  }

  SECTION("Take ownership of native handle")
  {
    ::cudaMemPoolProps pool_properties{};
    pool_properties.handleTypes = ::cudaMemAllocationHandleType(cudaMemAllocationHandleType::cudaMemHandleTypeNone);
    if (cuda::std::is_same_v<memory_pool, cudax::device_memory_pool>)
    {
      pool_properties.allocType     = ::cudaMemAllocationTypePinned;
      pool_properties.location.type = ::cudaMemLocationTypeDevice;
      pool_properties.location.id   = current_device;
    }
#if _CCCL_CUDACC_AT_LEAST(12, 6)
    else if (cuda::std::is_same_v<memory_pool, cudax::pinned_memory_pool>)
    {
      pool_properties.allocType     = ::cudaMemAllocationTypePinned;
      pool_properties.location.type = cudaMemLocationTypeHostNuma;
      pool_properties.location.id   = 0;
    }
#  if _CCCL_CUDACC_AT_LEAST(13, 0)
    else if (cuda::std::is_same_v<memory_pool, cudax::managed_memory_pool>)
    {
      pool_properties.allocType     = ::cudaMemAllocationTypeManaged;
      pool_properties.location.type = cudaMemLocationTypeNone;
      pool_properties.location.id   = 0;
    }
#  endif
#endif
    else
    {
      REQUIRE(false);
    }
    ::cudaMemPool_t new_pool{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &new_pool, &pool_properties);

    memory_pool from_handle = memory_pool::from_native_handle(new_pool);
    CHECK(from_handle.get() == new_pool);
  }
}

C2H_CCCLRT_TEST_LIST("device_memory_pool comparison", "[memory_resource]", TEST_TYPES)
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with with cudaGetDevice.", &current_device);
  }

  int driver_version = 0;
  {
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &driver_version);
  }

  ::cudaMemPool_t current_default_pool{};
  {
    _CCCL_TRY_CUDA_API(::cudaDeviceGetDefaultMemPool,
                       "Failed to call cudaDeviceGetDefaultMemPool",
                       &current_default_pool,
                       current_device);
  }

  using memory_pool = TestType;
  memory_pool first = construct_pool<memory_pool>(current_device);
  { // comparison against a plain device_memory_pool
    memory_pool second = construct_pool<memory_pool>(current_device);
    CHECK(first == first);
    CHECK(first != second);
  }
}

C2H_CCCLRT_TEST_LIST("device_memory_pool accessors", "[memory_resource]", TEST_TYPES)
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with with cudaGetDevice.", &current_device);
  }

  int driver_version = 0;
  {
    _CCCL_TRY_CUDA_API(::cudaDriverGetVersion, "Failed to call cudaDriverGetVersion", &driver_version);
  }

  using memory_pool     = TestType;
  using memory_resource = typename memory_pool::reference_type;
  SECTION("device_memory_pool::set_attribute")
  {
    memory_pool pool = construct_pool<memory_pool>(current_device);

    { // cudaMemPoolReuseFollowEventDependencies
      // Get the attribute value
      bool attr = pool.attribute(cudax::memory_pool_attributes::reuse_follow_event_dependencies);

      // Set it to the opposite
      pool.set_attribute(cudax::memory_pool_attributes::reuse_follow_event_dependencies, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.attribute(cudax::memory_pool_attributes::reuse_follow_event_dependencies);
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(cudax::memory_pool_attributes::reuse_follow_event_dependencies, attr);
    }

    { // cudaMemPoolReuseAllowOpportunistic
      // Get the attribute value
      bool attr = pool.attribute(cudax::memory_pool_attributes::reuse_allow_opportunistic);

      // Set it to the opposite
      pool.set_attribute(cudax::memory_pool_attributes::reuse_allow_opportunistic, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.attribute(cudax::memory_pool_attributes::reuse_allow_opportunistic);
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(cudax::memory_pool_attributes::reuse_allow_opportunistic, attr);
    }

    { // cudaMemPoolReuseAllowInternalDependencies
      // Get the attribute value
      bool attr = pool.attribute(cudax::memory_pool_attributes::reuse_allow_internal_dependencies);

      // Set it to the opposite
      pool.set_attribute(cudax::memory_pool_attributes::reuse_allow_internal_dependencies, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.attribute(cudax::memory_pool_attributes::reuse_allow_internal_dependencies);
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(cudax::memory_pool_attributes::reuse_allow_internal_dependencies, attr);
    }

    { // cudaMemPoolAttrReleaseThreshold
      // Get the attribute value
      size_t attr = pool.attribute(cudax::memory_pool_attributes::release_threshold);

      // Set it to something else
      pool.set_attribute(cudax::memory_pool_attributes::release_threshold, 2 * attr);

      // Retrieve again and verify it was changed
      size_t new_attr = pool.attribute(cudax::memory_pool_attributes::release_threshold);
      CHECK(new_attr == 2 * attr);

      // Set it back
      pool.set_attribute(cudax::memory_pool_attributes::release_threshold, attr);
    }

    // prime the pool to a given size
    memory_resource resource{pool};
    cudax::stream stream{cuda::device_ref{0}};

    // Allocate a buffer to prime
    auto* ptr = resource.allocate(stream, 256 * sizeof(int));
    stream.sync();

    { // cudaMemPoolAttrReservedMemHigh
      // Get the attribute value
      size_t attr = pool.attribute(cudax::memory_pool_attributes::reserved_mem_high);

      // Set it to zero as everything else is illegal
      pool.set_attribute(cudax::memory_pool_attributes::reserved_mem_high, 0);

      // Retrieve again and verify it was changed, which it wasn't...
      size_t new_attr = pool.attribute(cudax::memory_pool_attributes::reserved_mem_high);
      CHECK(new_attr == attr);

#if _CCCL_HAS_EXCEPTIONS()
      try
      {
        // Ensure we catch the contract violation
        pool.set_attribute(cudax::memory_pool_attributes::reserved_mem_high, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(), "This attribute can't be set to a non-zero value.") == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
#endif // _CCCL_HAS_EXCEPTIONS()
    }

    { // cudaMemPoolAttrUsedMemHigh
      // Get the attribute value
      size_t attr = pool.attribute(cudax::memory_pool_attributes::used_mem_high);

      // Set it to zero as everything else is illegal
      pool.set_attribute(cudax::memory_pool_attributes::used_mem_high, 0);

      // Retrieve again and verify it was changed, which it wasn't...
      size_t new_attr = pool.attribute(cudax::memory_pool_attributes::used_mem_high);
      CHECK(new_attr == attr);

#if _CCCL_HAS_EXCEPTIONS()
      try
      {
        // Ensure we catch the contract violation
        pool.set_attribute(cudax::memory_pool_attributes::used_mem_high, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(), "This attribute can't be set to a non-zero value.") == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
#endif // _CCCL_HAS_EXCEPTIONS()
    }

    // Reallocate as the checks above have screwed with the allocation count
    resource.deallocate(stream, ptr, 256 * sizeof(int));
    ptr = resource.allocate(stream, 2048 * sizeof(int));
    stream.sync();

    { // cudaMemPoolAttrReservedMemCurrent
      // Get the attribute value
      size_t attr = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
      CHECK(attr >= 2048 * sizeof(int));
      // cudaMemPoolAttrReservedMemCurrent cannot be set
    }

    { // cudaMemPoolAttrUsedMemCurrent
      // Get the attribute value
      size_t attr = pool.attribute(cudax::memory_pool_attributes::used_mem_current);
      CHECK(attr == 2048 * sizeof(int));
      // cudaMemPoolAttrUsedMemCurrent cannot be set
    }

    // Free the last allocation
    resource.deallocate(stream, ptr, 2048 * sizeof(int));
    stream.sync();
  }

  SECTION("device_memory_pool::trim_to")
  {
    memory_pool pool = construct_pool<memory_pool>(current_device);

    // prime the pool to a given size
    memory_resource resource{pool};
    cudax::stream stream{cuda::device_ref{0}};

    // Allocate 2 buffers
    auto* ptr1 = resource.allocate(stream, 2048 * sizeof(int));
    auto* ptr2 = resource.allocate(stream, 2048 * sizeof(int));
    resource.deallocate(stream, ptr1, 2048 * sizeof(int));
    stream.sync();

    // Ensure that we still hold some memory, otherwise everything is freed
    auto backing_size = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(backing_size >= 4096 * sizeof(int));

    // Trim the pool to something smaller than currently held
    pool.trim_to(1024);

    // Should be a noop
    auto noop_backing_size = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(backing_size == noop_backing_size);

    // Trim to larger than ever allocated
    pool.trim_to(backing_size * 24);

    // Should be a noop
    auto another_noop_backing_size = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(backing_size == another_noop_backing_size);

    // Trim to smaller than current backing but larger than current allocated
    pool.trim_to(2560 * sizeof(int));

    // Check the backing size again
    auto new_backing_size = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(new_backing_size <= backing_size);
    CHECK(new_backing_size >= 4096 * sizeof(int));

    // Free the last allocation
    resource.deallocate(stream, ptr2, 2048 * sizeof(int));
    stream.sync();

    // By default the pool should not release anything without a trim call
    auto no_backing = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(no_backing == new_backing_size);

    // We can still trim the pool without effect
    pool.trim_to(2560 * sizeof(int));

    auto still_no_backing = pool.attribute(cudax::memory_pool_attributes::reserved_mem_current);
    CHECK(still_no_backing == new_backing_size);
  }
}

C2H_CCCLRT_TEST("device_memory_pool::enable_access", "[memory_resource]")
{
  if (cuda::devices.size() > 1)
  {
    auto peers = cuda::devices[0].peers();
    if (peers.size() > 0)
    {
      cudax::device_memory_pool pool{cuda::devices[0]};
      CUDAX_CHECK(pool.is_accessible_from(cuda::devices[0]));

      pool.enable_access_from(peers);
      CUDAX_CHECK(pool.is_accessible_from(peers.front()));

      pool.disable_access_from(peers.front());
      CUDAX_CHECK(!pool.is_accessible_from(peers.front()));

      if (peers.size() > 1)
      {
        CUDAX_CHECK(pool.is_accessible_from(peers[1]));
      }
    }
  }
}

#if _CCCL_CUDACC_AT_LEAST(12, 6)
C2H_CCCLRT_TEST("pinned_memory_pool::enable_access", "[memory_resource]")
{
  cudax::pinned_memory_pool pool{0};
  CUDAX_CHECK(pool.is_accessible_from(cuda::devices[0]));

  // Currently bugged, need to wait for driver fix
  // pool.disable_access_from(cuda::devices[0]);
  // CUDAX_CHECK(!pool.is_accessible_from(cuda::devices[0]));

  // pool.enable_access_from(cuda::devices[0]);
  // CUDAX_CHECK(pool.is_accessible_from(cuda::devices[0]));
}
#endif

C2H_CCCLRT_TEST("device_memory_pool with allocation handle", "[memory_resource]")
{
  cudax::memory_pool_properties props              = {20, 42, ::cudaMemHandleTypePosixFileDescriptor};
  cudax::device_memory_pool with_allocation_handle = cudax::device_memory_pool(cuda::device_ref{0}, props);

  ::cudaMemPool_t current_default_pool{};
  {
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceGetDefaultMemPool, "Failed to call cudaDeviceGetDefaultMemPool", &current_default_pool, 0);
  }

  ::cudaMemPool_t get = with_allocation_handle.get();
  CHECK(get != current_default_pool);

  // Ensure we use the right release threshold
  CHECK(ensure_release_threshold(get, props.release_threshold));

  // Ensure that we disable reuse
  CHECK(ensure_disable_reuse(get));

  // Ensure that we disable export
  CHECK(ensure_export_handle(get, static_cast<cudaMemAllocationHandleType>(props.allocation_handle_type)));
}

#if _CCCL_CUDACC_AT_LEAST(12, 6)
C2H_CCCLRT_TEST("pinned_memory_pool with allocation handle", "[memory_resource]")
{
  cudax::memory_pool_properties props              = {20, 42, ::cudaMemHandleTypePosixFileDescriptor};
  cudax::pinned_memory_pool with_allocation_handle = cudax::pinned_memory_pool(0, props);

  ::cudaMemPool_t get = with_allocation_handle.get();
  CHECK(get != cudax::pinned_memory_resource{}.get());

  // Ensure we use the right release threshold
  CHECK(ensure_release_threshold(get, props.release_threshold));

  // Ensure that we disable reuse
  CHECK(ensure_disable_reuse(get));

  // Ensure that we disable export
  CHECK(ensure_export_handle(get, static_cast<cudaMemAllocationHandleType>(props.allocation_handle_type)));
}
#endif // _CCCL_CUDACC_AT_LEAST(12, 6)

// managed memory pool does not support allocation handles yet.
