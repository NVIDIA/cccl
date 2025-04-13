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

namespace cudax = cuda::experimental;

using managed_resource = cudax::managed_memory_resource;
static_assert(!cuda::std::is_trivial<managed_resource>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<managed_resource>::value, "");
static_assert(cuda::std::is_trivially_copy_constructible<managed_resource>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<managed_resource>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<managed_resource>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<managed_resource>::value, "");
static_assert(cuda::std::is_trivially_destructible<managed_resource>::value, "");
static_assert(!cuda::std::is_empty<managed_resource>::value, "");

static void ensure_managed_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeManaged);
}

C2H_TEST("managed_memory_resource construction", "[memory_resource]")
{
  SECTION("Default construction")
  {
    STATIC_REQUIRE(cuda::std::is_default_constructible_v<managed_resource>);
  }

  SECTION("Construct with flag")
  {
    managed_resource defaulted{};
    managed_resource with_flag{cudaMemAttachHost};
    CHECK(defaulted != with_flag);
  }
}

C2H_TEST("managed_memory_resource allocation", "[memory_resource]")
{
  managed_resource res{};
  cudax::stream stream{};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_managed_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_managed_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }

  { // allocate_async / deallocate_async
    auto* ptr = res.allocate_async(42, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.sync();
    ensure_managed_ptr(ptr);

    res.deallocate_async(ptr, 42, stream);
  }

  { // allocate_async / deallocate_async with alignment
    auto* ptr = res.allocate_async(42, 4, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.sync();
    ensure_managed_ptr(ptr);

    res.deallocate_async(ptr, 42, 4, stream);
  }

#if _CCCL_HAS_EXCEPTIONS()
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        [[maybe_unused]] auto* ptr = res.allocate(5, 42);
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
        [[maybe_unused]] auto* ptr = res.allocate(5, 1337);
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
      try
      {
        [[maybe_unused]] auto* ptr = res.allocate_async(5, 42, stream);
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
    }
  }

  { // allocate_async with non matching alignment
    while (true)
    {
      try
      {
        [[maybe_unused]] auto* ptr = res.allocate_async(5, 1337, stream);
      }
      catch (std::invalid_argument&)
      {
        break;
      }
      CHECK(false);
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
  void* allocate(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(void*, size_t, size_t) noexcept {}

  bool operator==(const resource&) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }
};
static_assert(cuda::mr::resource<resource<AccessibilityType::Host>>, "");
static_assert(cuda::mr::resource<resource<AccessibilityType::Device>>, "");

template <AccessibilityType Accessibility>
struct async_resource : public resource<Accessibility>
{
  void* allocate_async(size_t, size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, size_t, size_t, cuda::stream_ref) {}
};
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Host>>, "");
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Device>>, "");

// test for cccl#2214: https://github.com/NVIDIA/cccl/issues/2214
struct derived_managed_resource : cudax::managed_memory_resource
{
  using cudax::managed_memory_resource::managed_memory_resource;
};
static_assert(cuda::mr::resource<derived_managed_resource>, "");

C2H_TEST("managed_memory_resource comparison", "[memory_resource]")
{
  managed_resource first{};
  { // comparison against a plain managed_memory_resource
    managed_resource second{};
    CHECK((first == second));
    CHECK(!(first != second));
  }

  { // comparison against a plain managed_memory_resource with a different pool
    managed_resource second{cudaMemAttachHost};
    CHECK((first != second));
    CHECK(!(first == second));
  }

  { // comparison against a managed_memory_resource wrapped inside a resource_ref<device_accessible>
    managed_resource second{};
    cudax::resource_ref<cudax::device_accessible> second_ref{second};
    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a managed_memory_resource wrapped inside a async_resource_ref
    managed_resource second{};
    cudax::async_resource_ref<cudax::device_accessible> second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different managed_resource through resource_ref
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

  { // comparison against a different managed_resource through resource_ref
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
