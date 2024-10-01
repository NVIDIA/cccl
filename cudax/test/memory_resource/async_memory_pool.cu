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

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <stdexcept>

#include <catch2/catch.hpp>

namespace cudax = cuda::experimental;
using pool      = cudax::mr::async_memory_pool;
static_assert(!cuda::std::is_trivial<pool>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<pool>::value, "");
static_assert(!cuda::std::is_default_constructible<pool>::value, "");
static_assert(!cuda::std::is_copy_constructible<pool>::value, "");
static_assert(!cuda::std::is_move_constructible<pool>::value, "");
static_assert(!cuda::std::is_copy_assignable<pool>::value, "");
static_assert(!cuda::std::is_move_assignable<pool>::value, "");
static_assert(!cuda::std::is_trivially_destructible<pool>::value, "");
static_assert(!cuda::std::is_empty<pool>::value, "");

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

static bool ensure_disable_reuse(::cudaMemPool_t pool, const int driver_version)
{
  int disable_reuse = 0;
  _CCCL_TRY_CUDA_API(
    ::cudaMemPoolGetAttribute,
    "Failed to call cudaMemPoolGetAttribute",
    pool,
    ::cudaMemPoolReuseAllowOpportunistic,
    &disable_reuse);

  constexpr int min_async_version = 11050;
  return driver_version < min_async_version ? disable_reuse == 0 : disable_reuse != 0;
}

static bool ensure_export_handle(::cudaMemPool_t pool, const ::cudaMemAllocationHandleType allocation_handle)
{
  size_t handle              = 0;
  const ::cudaError_t status = ::cudaMemPoolExportToShareableHandle(&handle, pool, allocation_handle, 0);
  ::cudaGetLastError(); // Clear CUDA error state

  // If no export was defined we need to querry cudaErrorInvalidValue
  return allocation_handle == ::cudaMemHandleTypeNone ? status == ::cudaErrorInvalidValue : status == ::cudaSuccess;
}

TEST_CASE("async_memory_pool construction", "[memory_resource]")
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

  using memory_pool = cudax::mr::async_memory_pool;
  SECTION("Construct from device id")
  {
    cudax::mr::async_memory_pool from_device{current_device};

    ::cudaMemPool_t get = from_device.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, 0));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with empty properties")
  {
    cudax::mr::async_memory_pool_properties props{};
    memory_pool from_defaulted_properties{current_device, props};

    ::cudaMemPool_t get = from_defaulted_properties.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, 0));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with initial pool size")
  {
    cudax::mr::async_memory_pool_properties props = {42, 20};
    memory_pool with_threshold{current_device, props};

    ::cudaMemPool_t get = with_threshold.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  // Allocation handles are only supported after 11.2
#if !defined(_CCCL_CUDACC_BELOW_11_2)
  SECTION("Construct with allocation handle")
  {
    cudax::mr::async_memory_pool_properties props = {
      42, 20, cudax::mr::cudaMemAllocationHandleType::cudaMemHandleTypePosixFileDescriptor};
    memory_pool with_allocation_handle{current_device, props};

    ::cudaMemPool_t get = with_allocation_handle.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, static_cast<cudaMemAllocationHandleType>(props.allocation_handle_type)));
  }
#endif // !_CCCL_CUDACC_BELOW_11_2

  SECTION("Take ownership of native handle")
  {
    ::cudaMemPoolProps pool_properties{};
    pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    pool_properties.handleTypes   = ::cudaMemAllocationHandleType(cudaMemAllocationHandleType::cudaMemHandleTypeNone);
    pool_properties.location.type = ::cudaMemLocationTypeDevice;
    pool_properties.location.id   = current_device;
    ::cudaMemPool_t new_pool{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &new_pool, &pool_properties);

    cudax::mr::async_memory_pool from_handle = cudax::mr::async_memory_pool::from_native_handle(new_pool);
    CHECK(from_handle.get() == new_pool);
  }
}

TEST_CASE("async_memory_pool comparison", "[memory_resource]")
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to querry current device with with cudaGetDevice.", &current_device);
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

  cudax::mr::async_memory_pool first{current_device};
  { // comparison against a plain async_memory_pool
    cudax::mr::async_memory_pool second{current_device};
    CHECK(first == first);
    CHECK(first != second);
  }

  { // comparison against a cudaMemPool_t
    CHECK(first == first.get());
    CHECK(first.get() == first);
    CHECK(first != current_default_pool);
    CHECK(current_default_pool != first);
  }
}

TEST_CASE("async_memory_pool accessors", "[memory_resource]")
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to querry current device with with cudaGetDevice.", &current_device);
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

  SECTION("async_memory_pool::set_attribute")
  {
    cudax::mr::async_memory_pool pool{current_device};

    { // cudaMemPoolReuseFollowEventDependencies
      // Get the attribute value
      bool attr = pool.get_attribute(::cudaMemPoolReuseFollowEventDependencies) != 0;

      // Set it to the opposite
      pool.set_attribute(::cudaMemPoolReuseFollowEventDependencies, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.get_attribute(::cudaMemPoolReuseFollowEventDependencies) != 0;
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(::cudaMemPoolReuseFollowEventDependencies, attr);
    }

    { // cudaMemPoolReuseAllowOpportunistic
      // Get the attribute value
      bool attr = pool.get_attribute(::cudaMemPoolReuseAllowOpportunistic) != 0;

      // Set it to the opposite
      pool.set_attribute(::cudaMemPoolReuseAllowOpportunistic, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.get_attribute(::cudaMemPoolReuseAllowOpportunistic) != 0;
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(::cudaMemPoolReuseAllowOpportunistic, attr);
    }

    { // cudaMemPoolReuseAllowInternalDependencies
      // Get the attribute value
      bool attr = pool.get_attribute(::cudaMemPoolReuseAllowInternalDependencies) != 0;

      // Set it to the opposite
      pool.set_attribute(::cudaMemPoolReuseAllowInternalDependencies, !attr);

      // Retrieve again and verify it was changed
      bool new_attr = pool.get_attribute(::cudaMemPoolReuseAllowInternalDependencies) != 0;
      CHECK(attr == !new_attr);

      // Set it back
      pool.set_attribute(::cudaMemPoolReuseAllowInternalDependencies, attr);
    }

    { // cudaMemPoolAttrReleaseThreshold
      // Get the attribute value
      size_t attr = pool.get_attribute(::cudaMemPoolAttrReleaseThreshold);

      // Set it to something else
      pool.set_attribute(::cudaMemPoolAttrReleaseThreshold, 2 * attr);

      // Retrieve again and verify it was changed
      size_t new_attr = pool.get_attribute(::cudaMemPoolAttrReleaseThreshold);
      CHECK(new_attr == 2 * attr);

      // Set it back
      pool.set_attribute(::cudaMemPoolAttrReleaseThreshold, attr);
    }

    // prime the pool to a given size
    cudax::mr::async_memory_resource resource{pool};
    cudax::stream stream{};

    // Allocate a buffer to prime
    auto* ptr = resource.allocate_async(256 * sizeof(int), stream);
    stream.wait();

    { // cudaMemPoolAttrReservedMemHigh
      // Get the attribute value
      size_t attr = pool.get_attribute(::cudaMemPoolAttrReservedMemHigh);

      // Set it to zero as everything else is illegal
      pool.set_attribute(::cudaMemPoolAttrReservedMemHigh, 0);

      // Retrieve again and verify it was changed, which it wasnt...
      size_t new_attr = pool.get_attribute(::cudaMemPoolAttrReservedMemHigh);
      CHECK(new_attr == attr);

      try
      {
        // Ensure we catch the contract violation
        pool.set_attribute(::cudaMemPoolAttrReservedMemHigh, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(),
                     "async_memory_pool::set_attribute: It is illegal to set this attribute to a non-zero value.")
              == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
    }

    { // cudaMemPoolAttrUsedMemHigh
      // Get the attribute value
      size_t attr = pool.get_attribute(::cudaMemPoolAttrUsedMemHigh);

      // Set it to zero as everything else is illegal
      pool.set_attribute(::cudaMemPoolAttrUsedMemHigh, 0);

      // Retrieve again and verify it was changed, which it wasnt...
      size_t new_attr = pool.get_attribute(::cudaMemPoolAttrUsedMemHigh);
      CHECK(new_attr == attr);

      try
      {
        // Ensure we catch the contract violation
        pool.set_attribute(::cudaMemPoolAttrUsedMemHigh, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(),
                     "async_memory_pool::set_attribute: It is illegal to set this attribute to a non-zero value.")
              == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
    }

    // Reallocate as the checks above have screwed with the allocation count
    resource.deallocate_async(ptr, 256 * sizeof(int), stream);
    ptr = resource.allocate_async(2048 * sizeof(int), stream);
    stream.wait();

    { // cudaMemPoolAttrReservedMemCurrent
      // Get the attribute value
      size_t attr = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
      CHECK(attr >= 2048 * sizeof(int));
      // cudaMemPoolAttrReservedMemCurrent cannot be set
      try
      {
        pool.set_attribute(::cudaMemPoolAttrReservedMemCurrent, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(), "Invalid attribute passed to async_memory_pool::set_attribute.") == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
    }

    { // cudaMemPoolAttrUsedMemCurrent
      // Get the attribute value
      size_t attr = pool.get_attribute(::cudaMemPoolAttrUsedMemCurrent);
      CHECK(attr == 2048 * sizeof(int));
      // cudaMemPoolAttrUsedMemCurrent cannot be set
      try
      {
        pool.set_attribute(::cudaMemPoolAttrUsedMemCurrent, attr);
        CHECK(false);
      }
      catch (::std::invalid_argument& err)
      {
        CHECK(strcmp(err.what(), "Invalid attribute passed to async_memory_pool::set_attribute.") == 0);
      }
      catch (...)
      {
        CHECK(false);
      }
    }

    // Free the last allocation
    resource.deallocate_async(ptr, 2048 * sizeof(int), stream);
    stream.wait();
  }

  SECTION("async_memory_pool::trim_to")
  {
    cudax::mr::async_memory_pool pool{current_device};
    // prime the pool to a given size
    cudax::mr::async_memory_resource resource{pool};
    cudax::stream stream{};

    // Allocate 2 buffers
    auto* ptr1 = resource.allocate_async(2048 * sizeof(int), stream);
    auto* ptr2 = resource.allocate_async(2048 * sizeof(int), stream);
    resource.deallocate_async(ptr1, 2048 * sizeof(int), stream);
    stream.wait();

    // Ensure that we still hold some memory, otherwise everything is freed
    auto backing_size = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(backing_size >= 4096 * sizeof(int));

    // Trim the pool to something smaller than currently held
    pool.trim_to(1024);

    // Should be a noop
    auto noop_backing_size = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(backing_size == noop_backing_size);

    // Trim to larger than ever allocated
    pool.trim_to(backing_size * 24);

    // Should be a noop
    auto another_noop_backing_size = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(backing_size == another_noop_backing_size);

    // Trim to smaller than current backing but larger than current allocated
    pool.trim_to(2560 * sizeof(int));

    // Check the backing size again
    auto new_backing_size = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(new_backing_size <= backing_size);
    CHECK(new_backing_size >= 4096 * sizeof(int));

    // Free the last allocation
    resource.deallocate_async(ptr2, 2048 * sizeof(int), stream);
    stream.wait();

    // There is nothing allocated anymore, so all memory is released
    auto no_backing = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(no_backing == 0);

    // We can still trim the pool without effect
    pool.trim_to(2560 * sizeof(int));

    auto still_no_backing = pool.get_attribute(::cudaMemPoolAttrReservedMemCurrent);
    CHECK(still_no_backing == 0);
  }
}
