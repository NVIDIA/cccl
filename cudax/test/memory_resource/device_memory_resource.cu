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

#include <stdexcept>

#include <catch2/catch.hpp>
#include <utility.cuh>

namespace cudax = cuda::experimental;

static_assert(!cuda::std::is_trivial<cudax::device_memory_resource>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_default_constructible<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_copy_constructible<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_move_constructible<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_copy_assignable<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_move_assignable<cudax::device_memory_resource>::value, "");
static_assert(cuda::std::is_trivially_destructible<cudax::device_memory_resource>::value, "");
static_assert(!cuda::std::is_empty<cudax::device_memory_resource>::value, "");

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

  // If no export was defined we need to query cudaErrorInvalidValue
  return allocation_handle == ::cudaMemHandleTypeNone ? status == ::cudaErrorInvalidValue : status == ::cudaSuccess;
}

TEST_CASE("device_memory_resource construction", "[memory_resource]")
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with cudaGetDevice.", &current_device);
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

  using async_resource = cuda::experimental::device_memory_resource;
  SECTION("Default construction")
  {
    {
      async_resource default_constructed{};
      CHECK(default_constructed.get() == current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::experimental::device_memory_resource",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    CHECK(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync,
      "Failed to deallocate with pool passed to cuda::experimental::device_memory_resource",
      ptr,
      ::cudaStream_t{0});
  }

  SECTION("Construct from mempool handle")
  {
    ::cudaMemPoolProps pool_properties{};
    pool_properties.allocType     = ::cudaMemAllocationTypePinned;
    pool_properties.handleTypes   = ::cudaMemAllocationHandleType(0);
    pool_properties.location.type = ::cudaMemLocationTypeDevice;
    pool_properties.location.id   = current_device;
    cudaMemPool_t cuda_pool_handle{};
    _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &cuda_pool_handle, &pool_properties);

    {
      async_resource from_cudaMemPool{cuda_pool_handle};
      CHECK(from_cudaMemPool.get() == cuda_pool_handle);
      CHECK(from_cudaMemPool.get() != current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::experimental::device_memory_resource",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    CHECK(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync,
      "Failed to deallocate with pool passed to cuda::experimental::device_memory_resource",
      ptr,
      ::cudaStream_t{0});
  }

  SECTION("Construct with initial pool size")
  {
    cuda::experimental::memory_pool_properties props = {
      42,
    };
    cuda::experimental::device_memory_pool pool{current_device, props};
    async_resource from_initial_pool_size{pool};

    ::cudaMemPool_t get = from_initial_pool_size.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, 0));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with release threshold")
  {
    cuda::experimental::memory_pool_properties props = {
      42,
      20,
    };
    cuda::experimental::device_memory_pool pool{current_device, props};
    async_resource with_threshold{pool};

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
#if _CCCL_CUDACC_AT_LEAST(11, 2)
  SECTION("Construct with allocation handle")
  {
    cuda::experimental::memory_pool_properties props = {
      42,
      20,
      cuda::experimental::cudaMemAllocationHandleType::cudaMemHandleTypePosixFileDescriptor,
    };
    cuda::experimental::device_memory_pool pool{current_device, props};
    async_resource with_allocation_handle{pool};

    ::cudaMemPool_t get = with_allocation_handle.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, static_cast<cudaMemAllocationHandleType>(props.allocation_handle_type)));
  }
#endif // _CCCL_CUDACC_AT_LEAST(11, 2)
}

static void ensure_device_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeDevice);
}

TEST_CASE("device_memory_resource allocation", "[memory_resource]")
{
  cuda::experimental::device_memory_resource res{};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }

  { // allocate_async / deallocate_async
    cudaStream_t raw_stream;
    cudaStreamCreate(&raw_stream);
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate_async(42, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_device_ptr(ptr);

    res.deallocate_async(ptr, 42, stream);
    cudaStreamDestroy(raw_stream);
  }

  { // allocate_async / deallocate_async with alignment
    cudaStream_t raw_stream;
    cudaStreamCreate(&raw_stream);
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate_async(42, 4, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_device_ptr(ptr);

    res.deallocate_async(ptr, 42, 4, stream);
    cudaStreamDestroy(raw_stream);
  }

#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 42);
        (void) ptr;
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
    }
  }

  { // allocate with non matching alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 1337);
        (void) ptr;
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
    }
  }
  { // allocate_async with too small alignment
    while (true)
    {
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      try
      {
        auto* ptr = res.allocate_async(5, 42, raw_stream);
        (void) ptr;
      }
      catch (std::invalid_argument&)
      {
        cudaStreamDestroy(raw_stream);
        break;
      }
      CHECK(false);
    }
  }

  { // allocate_async with non matching alignment
    while (true)
    {
      cudaStream_t raw_stream;
      cudaStreamCreate(&raw_stream);
      try
      {
        auto* ptr = res.allocate_async(5, 1337, raw_stream);
        (void) ptr;
      }
      catch (std::invalid_argument&)
      {
        cudaStreamDestroy(raw_stream);
        break;
      }
      CHECK(false);
    }
  }
#endif // _LIBCUDACXX_NO_EXCEPTIONS
}

enum class AccessibilityType
{
  Device,
  Host,
};

template <AccessibilityType Accessibilty>
struct resource
{
  void* allocate(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(void*, size_t, size_t) {}

  bool operator==(const resource&) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }

  template <AccessibilityType Accessibilty2                                         = Accessibilty,
            cuda::std::enable_if_t<Accessibilty2 == AccessibilityType::Device, int> = 0>
  friend void get_property(const resource&, cuda::mr::device_accessible) noexcept
  {}
};
static_assert(cuda::mr::resource<resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::resource_with<resource<AccessibilityType::Host>, cuda::mr::device_accessible>, "");
static_assert(cuda::mr::resource<resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::resource_with<resource<AccessibilityType::Device>, cuda::mr::device_accessible>, "");

template <AccessibilityType Accessibilty>
struct async_resource : public resource<Accessibilty>
{
  void* allocate_async(size_t, size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, size_t, size_t, cuda::stream_ref) {}
};
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::async_resource_with<async_resource<AccessibilityType::Host>, cuda::mr::device_accessible>, "");
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::async_resource_with<async_resource<AccessibilityType::Device>, cuda::mr::device_accessible>,
              "");

TEST_CASE("device_memory_resource comparison", "[memory_resource]")
{
  int current_device{};
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device with cudaGetDevice.", &current_device);
  }

  cuda::experimental::device_memory_resource first{};
  { // comparison against a plain device_memory_resource
    cuda::experimental::device_memory_resource second{};
    CHECK(first == second);
    CHECK(!(first != second));
  }

  { // comparison against a plain device_memory_resource with a different pool
    cudaMemPool_t cuda_pool_handle{};
    {
      ::cudaMemPoolProps pool_properties{};
      pool_properties.allocType     = ::cudaMemAllocationTypePinned;
      pool_properties.handleTypes   = ::cudaMemAllocationHandleType(0);
      pool_properties.location.type = ::cudaMemLocationTypeDevice;
      pool_properties.location.id   = current_device;
      _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &cuda_pool_handle, &pool_properties);
    }
    cuda::experimental::device_memory_resource second{cuda_pool_handle};
    CHECK(first != second);
    CHECK(!(first == second));
  }

  { // comparison against a device_memory_resource wrapped inside a resource_ref<device_accessible>
    cuda::experimental::device_memory_resource second{};
    cuda::mr::resource_ref<cuda::mr::device_accessible> second_ref{second};
    CHECK(first == second_ref);
    CHECK(!(first != second_ref));
    CHECK(second_ref == first);
    CHECK(!(second_ref != first));
  }

  { // comparison against a device_memory_resource wrapped inside a async_resource_ref
    cuda::experimental::device_memory_resource second{};
    cuda::mr::async_resource_ref<cuda::mr::device_accessible> second_ref{second};

    CHECK(first == second_ref);
    CHECK(!(first != second_ref));
    CHECK(second_ref == first);
    CHECK(!(second_ref != first));
  }

  { // comparison against a different resource through resource_ref
    resource<AccessibilityType::Host> host_resource{};
    resource<AccessibilityType::Device> device_resource{};
    CHECK(!(first == host_resource));
    CHECK(first != host_resource);
    CHECK(!(first == device_resource));
    CHECK(first != device_resource);

    CHECK(!(host_resource == first));
    CHECK(host_resource != first);
    CHECK(!(device_resource == first));
    CHECK(device_resource != first);
  }

  { // comparison against a different resource through resource_ref
    async_resource<AccessibilityType::Host> host_async_resource{};
    async_resource<AccessibilityType::Device> device_async_resource{};
    CHECK(!(first == host_async_resource));
    CHECK(first != host_async_resource);
    CHECK(!(first == device_async_resource));
    CHECK(first != device_async_resource);

    CHECK(!(host_async_resource == first));
    CHECK(host_async_resource != first);
    CHECK(!(device_async_resource == first));
    CHECK(device_async_resource != first);
  }
}

TEST_CASE("Async memory resource peer access")
{
  if (cudax::devices.size() > 1)
  {
    auto peers = cudax::devices[0].get_peers();
    if (peers.size() > 0)
    {
      cudax::device_memory_pool pool{cudax::devices[0]};
      cudax::device_memory_resource resource{pool};
      cudax::stream stream{peers.front()};
      CUDAX_CHECK(resource.is_accessible_from(cudax::devices[0]));

      auto allocate_and_check_access = [&](auto& resource) {
        auto* ptr1 = resource.allocate_async(sizeof(int), stream);
        auto* ptr2 = resource.allocate(sizeof(int));
        auto dims  = cudax::distribute<1>(1);
        cudax::launch(stream, dims, test::assign_42{}, (int*) ptr1);
        cudax::launch(stream, dims, test::assign_42{}, (int*) ptr2);
        stream.wait();
        resource.deallocate_async(ptr1, sizeof(int), stream);
        resource.deallocate(ptr2, sizeof(int));
      };

      resource.enable_peer_access_from(peers);

      CUDAX_CHECK(pool.is_accessible_from(peers.front()));
      CUDAX_CHECK(resource.is_accessible_from(peers.front()));
      allocate_and_check_access(resource);

      cudax::device_memory_resource another_resource{pool};
      CUDAX_CHECK(another_resource.is_accessible_from(peers.front()));
      allocate_and_check_access(another_resource);

      resource.disable_peer_access_from(peers.front());
      CUDAX_CHECK(!resource.is_accessible_from(peers.front()));
      CUDAX_CHECK(!another_resource.is_accessible_from(peers.front()));

      if (peers.size() > 1)
      {
        CUDAX_CHECK(resource.is_accessible_from(peers[1]));
      }

      resource.disable_peer_access_from(peers);

      resource.enable_peer_access_from(peers.front());
      CUDAX_CHECK(resource.is_accessible_from(peers.front()));
      CUDAX_CHECK(another_resource.is_accessible_from(peers.front()));

      // Check if enable can include the device on which the pool resides
      peers.push_back(cudax::devices[0]);
      resource.enable_peer_access_from(peers);

      // Check the resource using the default pool
      cudax::device_memory_resource default_pool_resource{};
      cudax::device_memory_resource another_default_pool_resource{};

      default_pool_resource.enable_peer_access_from(peers.front());

      CUDAX_CHECK(default_pool_resource.is_accessible_from(peers.front()));
      allocate_and_check_access(default_pool_resource);
      CUDAX_CHECK(another_default_pool_resource.is_accessible_from(peers.front()));
      allocate_and_check_access(another_default_pool_resource);
    }
  }
}
