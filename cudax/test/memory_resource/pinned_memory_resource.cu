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

#include <testing.cuh>
#include <utility.cuh>

#include "common_tests.cuh"

namespace cudax = cuda::experimental;

#if _CCCL_CUDACC_AT_LEAST(12, 6)
#  define TEST_TYPES cudax::legacy_pinned_memory_resource, cudax::pinned_memory_resource
#else
#  define TEST_TYPES cudax::legacy_pinned_memory_resource
#endif

template <typename Resource>
void resource_static_asserts()
{
  static_assert(!cuda::std::is_trivial_v<Resource>, "");
  static_assert(!cuda::std::is_trivially_default_constructible_v<Resource>, "");
  static_assert(cuda::std::is_trivially_copy_constructible_v<Resource>, "");
  static_assert(cuda::std::is_trivially_move_constructible_v<Resource>, "");
  static_assert(cuda::std::is_trivially_copy_assignable_v<Resource>, "");
  static_assert(cuda::std::is_trivially_move_assignable_v<Resource>, "");
  static_assert(cuda::std::is_trivially_destructible_v<Resource>, "");
  static_assert(cuda::std::is_default_constructible_v<Resource>, "");
}

template void resource_static_asserts<cudax::legacy_pinned_memory_resource>();
#if _CCCL_CUDACC_AT_LEAST(12, 6)
template void resource_static_asserts<cudax::pinned_memory_resource>();
#endif

static void ensure_pinned_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeHost);
  // Driver bug fixed in r575
  // TODO Re-enable one we start testing with r575
  // CHECK(attributes.devicePointer != nullptr);
}

C2H_TEST_LIST("pinned_memory_resource allocation", "[memory_resource]", TEST_TYPES)
{
  using pinned_resource = TestType;
  pinned_resource res{};
  cudax::stream stream{cuda::device_ref{0}};

  { // allocate_sync / deallocate_sync
    auto* ptr = res.allocate_sync(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_ptr(ptr);

    res.deallocate_sync(ptr, 42);
  }

  { // allocate_sync / deallocate_sync with alignment
    auto* ptr = res.allocate_sync(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_ptr(ptr);

    res.deallocate_sync(ptr, 42, 4);
  }

  if constexpr (cuda::mr::resource<pinned_resource>)
  {
    { // allocate / deallocate
      auto* ptr = res.allocate(stream, 42);
      static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

      stream.sync();
      ensure_pinned_ptr(ptr);

      res.deallocate(stream, ptr, 42);
    }

    { // allocate / deallocate with alignment
      auto* ptr = res.allocate(stream, 42, 4);
      static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

      stream.sync();
      ensure_pinned_ptr(ptr);

      res.deallocate(stream, ptr, 42, 4);
    }
  }

#if _CCCL_HAS_EXCEPTIONS()
  { // allocate_sync with too small alignment
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

  { // allocate_sync with non matching alignment
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

  if constexpr (cuda::mr::resource<pinned_resource>)
  {
    { // allocate with too small alignment
      while (true)
      {
        try
        {
          [[maybe_unused]] auto* ptr = res.allocate(stream, 5, 42);
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
          auto* ptr = res.allocate(stream, 5, 1337);
          (void) ptr;
        }
        catch (std::invalid_argument&)
        {
          break;
        }
        CHECK(false);
      }
    }
  }
#endif // _CCCL_HAS_EXCEPTIONS()
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
  void deallocate_sync(void*, size_t, size_t) noexcept {}

  bool operator==(const resource&) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }
};
static_assert(cuda::mr::synchronous_resource<resource<AccessibilityType::Host>>, "");
static_assert(cuda::mr::synchronous_resource<resource<AccessibilityType::Device>>, "");

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
static_assert(cuda::mr::resource<test_resource<AccessibilityType::Device>>, "");

// test for cccl#2214: https://github.com/NVIDIA/cccl/issues/2214
struct derived_pinned_resource : cudax::legacy_pinned_memory_resource
{
  using legacy_pinned_memory_resource::legacy_pinned_memory_resource;
};
static_assert(cuda::mr::synchronous_resource<derived_pinned_resource>, "");

C2H_TEST_LIST("pinned_memory_resource comparison", "[memory_resource]", TEST_TYPES)
{
  using pinned_resource = TestType;
  pinned_resource first{};
  { // comparison against a plain pinned_memory_resource
    pinned_resource second{};
    CHECK((first == second));
    CHECK(!(first != second));
  }

  { // comparison against a pinned_memory_resource wrapped inside a synchronous_resource_ref<device_accessible>
    pinned_resource second{};
    cudax::synchronous_resource_ref<cudax::device_accessible> const second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  if constexpr (cuda::mr::resource<pinned_resource>)
  { // comparison against a pinned_memory_resource wrapped inside a resource_ref
    pinned_resource second{};
    cudax::resource_ref<cudax::device_accessible> second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different pinned_resource through synchronous_resource_ref
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

  { // comparison against a different pinned_resource through synchronous_resource_ref
    resource<AccessibilityType::Host> host_async_resource{};
    resource<AccessibilityType::Device> device_async_resource{};
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

#if _CCCL_CUDACC_AT_LEAST(12, 6)
C2H_TEST("pinned_memory_resource async.deallocate_sync", "[memory_resource]")
{
  cudax::pinned_memory_resource resource{};
  test_deallocate_async(resource);
}
#endif // _CCCL_CUDACC_AT_LEAST(12, 6)
