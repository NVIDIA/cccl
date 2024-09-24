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
