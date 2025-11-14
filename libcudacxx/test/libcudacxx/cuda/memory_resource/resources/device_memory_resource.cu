//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <stdexcept>

#include <testing.cuh>
#include <utility.cuh>

static_assert(!cuda::std::is_trivial<cuda::device_memory_pool_ref>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<cuda::device_memory_pool_ref>::value, "");
static_assert(cuda::std::is_copy_constructible<cuda::device_memory_pool_ref>::value, "");
static_assert(cuda::std::is_move_constructible<cuda::device_memory_pool_ref>::value, "");
static_assert(cuda::std::is_copy_assignable<cuda::device_memory_pool_ref>::value, "");
static_assert(cuda::std::is_move_assignable<cuda::device_memory_pool_ref>::value, "");
static_assert(cuda::std::is_trivially_destructible<cuda::device_memory_pool_ref>::value, "");
static_assert(!cuda::std::is_empty<cuda::device_memory_pool_ref>::value, "");

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

C2H_CCCLRT_TEST("device_memory_resource construction", "[memory_resource]")
{
  int current_device = 0;
  cuda::__ensure_current_context guard{cuda::device_ref{current_device}};

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

  using test_resource = cuda::device_memory_pool_ref;
  SECTION("Default construction")
  {
    {
      test_resource default_constructed = cuda::device_default_memory_pool(cuda::device_ref{0});
      CHECK(default_constructed.get() == current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::device_memory_pool_ref",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    CHECK(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, "Failed to deallocate with pool passed to cuda::device_memory_pool_ref", ptr, ::cudaStream_t{0});
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
      test_resource from_cudaMemPool{cuda_pool_handle};
      CHECK(from_cudaMemPool.get() == cuda_pool_handle);
      CHECK(from_cudaMemPool.get() != current_default_pool);
    }

    // Ensure that the pool was not destroyed by allocating something
    void* ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "Failed to allocate with pool passed to cuda::device_memory_pool_ref",
      &ptr,
      42,
      current_default_pool,
      ::cudaStream_t{0});
    CHECK(ptr != nullptr);

    _CCCL_ASSERT_CUDA_API(
      ::cudaFreeAsync, "Failed to deallocate with pool passed to cuda::device_memory_pool_ref", ptr, ::cudaStream_t{0});
  }

  SECTION("Construct with initial pool size")
  {
    cuda::memory_pool_properties props = {
      42,
    };
    cuda::device_memory_pool from_initial_pool_size{current_device, props};

    ::cudaMemPool_t get = from_initial_pool_size.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, cuda::std::numeric_limits<size_t>::max()));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, ::cudaMemHandleTypeNone));
  }

  SECTION("Construct with release threshold")
  {
    cuda::memory_pool_properties props = {
      20,
      42,
    };
    cuda::device_memory_pool with_threshold{current_device, props};

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
  SECTION("Construct with allocation handle")
  {
    cuda::memory_pool_properties props = {
      20,
      42,
      ::cudaMemHandleTypePosixFileDescriptor,
    };
    cuda::device_memory_pool with_allocation_handle{current_device, props};

    ::cudaMemPool_t get = with_allocation_handle.get();
    CHECK(get != current_default_pool);

    // Ensure we use the right release threshold
    CHECK(ensure_release_threshold(get, props.release_threshold));

    // Ensure that we disable reuse with unsupported drivers
    CHECK(ensure_disable_reuse(get, driver_version));

    // Ensure that we disable export
    CHECK(ensure_export_handle(get, props.allocation_handle_type));
  }
}

static void ensure_device_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeDevice);
}

C2H_CCCLRT_TEST("device_memory_resource allocation", "[memory_resource]")
{
  cudaStream_t raw_stream;
  {
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    cudaStreamCreate(&raw_stream);
  }
  cuda::device_memory_pool_ref res = cuda::device_default_memory_pool(cuda::device_ref{0});

  { // allocate_sync / deallocate_sync
    auto* ptr = res.allocate_sync(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate_sync(ptr, 42);
  }

  { // allocate_sync / deallocate_sync with alignment
    auto* ptr = res.allocate_sync(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate_sync(ptr, 42, 4);
  }

  { // allocate / deallocate
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate(stream, 42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.sync();
    ensure_device_ptr(ptr);

    res.deallocate(stream, ptr, 42);
  }

  { // allocate / deallocate with alignment
    cuda::stream_ref stream{raw_stream};

    auto* ptr = res.allocate(stream, 42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.sync();
    ensure_device_ptr(ptr);

    res.deallocate(stream, ptr, 42, 4);
  }

#if _CCCL_HAS_EXCEPTIONS()
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        [[maybe_unused]] auto* ptr = res.allocate_sync(5, 42);
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
        [[maybe_unused]] auto* ptr = res.allocate_sync(5, 1337);
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
    }
  }
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        [[maybe_unused]] auto* ptr = res.allocate(raw_stream, 5, 42);
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
        [[maybe_unused]] auto* ptr = res.allocate(raw_stream, 5, 1337);
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
    }
  }
#endif // _CCCL_HAS_EXCEPTIONS()
  {
    cuda::__ensure_current_context guard{cuda::device_ref{0}};
    cudaStreamDestroy(raw_stream);
  }
}

enum class AccessibilityType
{
  Device,
  Host,
};

template <AccessibilityType Accessibility>
struct resource
{
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) {}

  bool operator==(const resource&) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }

  template <AccessibilityType Accessibilty2                                         = Accessibility,
            cuda::std::enable_if_t<Accessibilty2 == AccessibilityType::Device, int> = 0>
  friend void get_property(const resource&, ::cuda::mr::device_accessible) noexcept
  {}
};
static_assert(cuda::mr::synchronous_resource<resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::synchronous_resource_with<resource<AccessibilityType::Host>, ::cuda::mr::device_accessible>,
              "");
static_assert(cuda::mr::synchronous_resource<resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::synchronous_resource_with<resource<AccessibilityType::Device>, ::cuda::mr::device_accessible>,
              "");

template <AccessibilityType Accessibility>
struct test_resource : public resource<Accessibility>
{
  void* allocate(cuda::stream_ref, size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, size_t, size_t) {}
};
static_assert(cuda::mr::resource<test_resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::resource_with<test_resource<AccessibilityType::Host>, ::cuda::mr::device_accessible>, "");
static_assert(cuda::mr::resource<test_resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::resource_with<test_resource<AccessibilityType::Device>, ::cuda::mr::device_accessible>, "");

C2H_CCCLRT_TEST("device_memory_resource comparison", "[memory_resource]")
{
  int current_device = 0;
  cuda::__ensure_current_context guard{cuda::device_ref{current_device}};

  cuda::device_memory_pool_ref first = cuda::device_default_memory_pool(cuda::device_ref{0});
  { // comparison against a plain device_memory_pool_ref
    cuda::device_memory_pool_ref second = cuda::device_default_memory_pool(cuda::device_ref{0});
    CHECK((first == second));
    CHECK(!(first != second));
  }

  { // comparison against a plain device_memory_pool_ref with a different pool
    cudaMemPool_t cuda_pool_handle{};
    {
      ::cudaMemPoolProps pool_properties{};
      pool_properties.allocType     = ::cudaMemAllocationTypePinned;
      pool_properties.handleTypes   = ::cudaMemAllocationHandleType(0);
      pool_properties.location.type = ::cudaMemLocationTypeDevice;
      pool_properties.location.id   = current_device;
      _CCCL_TRY_CUDA_API(::cudaMemPoolCreate, "Failed to call cudaMemPoolCreate", &cuda_pool_handle, &pool_properties);
    }
    cuda::device_memory_pool_ref second{cuda_pool_handle};
    CHECK((first != second));
    CHECK(!(first == second));
  }

  { // comparison against a device_memory_pool_ref wrapped inside a synchronous_resource_ref<device_accessible>
    cuda::device_memory_pool_ref second = cuda::device_default_memory_pool(cuda::device_ref{0});
    cuda::mr::synchronous_resource_ref<::cuda::mr::device_accessible> second_ref{second};
    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a device_memory_pool_ref wrapped inside a resource_ref
    cuda::device_memory_pool_ref second = cuda::device_default_memory_pool(cuda::device_ref{0});
    cuda::mr::resource_ref<::cuda::mr::device_accessible> second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different resource through synchronous_resource_ref
    resource<AccessibilityType::Host> host_resource{};
    resource<AccessibilityType::Device> device_resource{};
    CHECK(!(first == host_resource));
    CHECK((first != host_resource));
    CHECK(!(first == device_resource));
    CHECK((first != device_resource));

    CHECK(!(host_resource == first));
    CHECK((host_resource != first));
    CHECK(!(device_resource == first));
    CHECK((device_resource != first));
  }

  { // comparison against a different resource through synchronous_resource_ref
    test_resource<AccessibilityType::Host> host_async_resource{};
    test_resource<AccessibilityType::Device> device_async_resource{};
    CHECK(!(first == host_async_resource));
    CHECK((first != host_async_resource));
    CHECK(!(first == device_async_resource));
    CHECK((first != device_async_resource));

    CHECK(!(host_async_resource == first));
    CHECK((host_async_resource != first));
    CHECK(!(device_async_resource == first));
    CHECK((device_async_resource != first));
  }
}

C2H_CCCLRT_TEST("Async memory resource access", "")
{
  /* disable until we move the launch API to libcudacxx
  if (cuda::devices.size() > 1)
  {
    auto peers = cuda::devices[0].peers();
    if (peers.size() > 0)
    {
      cuda::device_memory_pool pool{cuda::devices[0]};
      cuda::stream_ref stream{peers.front()};
      CCCLRT_CHECK(pool.is_accessible_from(cuda::devices[0]));

      auto allocate_and_check_access = [&](auto& resource) {
        auto* ptr1  = resource.allocate(stream, sizeof(int));
        auto* ptr2  = resource.allocate_sync(sizeof(int));
        auto config = cudax::distribute<1>(1);
        cudax::launch(stream, config, test::assign_42{}, (int*) ptr1);
        cudax::launch(stream, config, test::assign_42{}, (int*) ptr2);
        stream.sync();
        resource.deallocate(stream, ptr1, sizeof(int));
        resource.deallocate_sync(ptr2, sizeof(int));
      };

      pool.enable_access_from(peers);

      CCCLRT_CHECK(pool.is_accessible_from(peers.front()));
      allocate_and_check_access(pool);

      cudax::device_memory_pool_ref resource{pool};
      CCCLRT_CHECK(resource.is_accessible_from(peers.front()));
      allocate_and_check_access(resource);

      pool.disable_access_from(peers.front());
      CCCLRT_CHECK(!pool.is_accessible_from(peers.front()));
      CCCLRT_CHECK(!resource.is_accessible_from(peers.front()));

      if (peers.size() > 1)
      {
        CCCLRT_CHECK(pool.is_accessible_from(peers[1]));
      }

      pool.disable_access_from(peers);

      pool.enable_access_from(peers.front());
      CCCLRT_CHECK(pool.is_accessible_from(peers.front()));
      CCCLRT_CHECK(resource.is_accessible_from(peers.front()));

      // Check if enable can include the device on which the pool resides
      {
        std::vector peers_ext(peers.begin(), peers.end());
        peers_ext.push_back(cuda::devices[0]);
        pool.enable_access_from(peers_ext);

        // Check the resource using the default pool
        cuda::device_memory_pool_ref default_pool_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
        cuda::device_memory_pool_ref another_default_pool_resource =
          cuda::device_default_memory_pool(cuda::device_ref{0});

        pool.enable_access_from(peers_ext.front());

        CCCLRT_CHECK(default_pool_resource.is_accessible_from(peers_ext.front()));
        allocate_and_check_access(default_pool_resource);
        CCCLRT_CHECK(another_default_pool_resource.is_accessible_from(peers_ext.front()));
        allocate_and_check_access(another_default_pool_resource);
      }
    }
  }
  */
}
