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

#include "cuda/__memory_resource/resource_ref.h"
#include <testing.cuh>
#include <utility.cuh>

namespace cudax = cuda::experimental;

using pinned_resource = cudax::pinned_memory_resource;
static_assert(!cuda::std::is_trivial<pinned_resource>::value, "");
static_assert(!cuda::std::is_trivially_default_constructible<pinned_resource>::value, "");
static_assert(cuda::std::is_trivially_copy_constructible<pinned_resource>::value, "");
static_assert(cuda::std::is_trivially_move_constructible<pinned_resource>::value, "");
static_assert(cuda::std::is_trivially_copy_assignable<pinned_resource>::value, "");
static_assert(cuda::std::is_trivially_move_assignable<pinned_resource>::value, "");
static_assert(cuda::std::is_trivially_destructible<pinned_resource>::value, "");
static_assert(!cuda::std::is_empty<pinned_resource>::value, "");

static void ensure_pinned_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeHost);
  CHECK(attributes.devicePointer != nullptr);
}

TEST_CASE("pinned_memory_resource construction", "[memory_resource]")
{
  SECTION("Default construction")
  {
    STATIC_REQUIRE(cuda::std::is_default_constructible_v<pinned_resource>);
  }

  SECTION("Construct with flag")
  {
    pinned_resource defaulted{};
    pinned_resource with_flag{cudaHostAllocMapped};
    CHECK(defaulted != with_flag);
  }
}

TEST_CASE("pinned_memory_resource allocation", "[memory_resource]")
{
  pinned_resource res{};
  cudax::stream stream{};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }

  { // allocate_async / deallocate_async
    auto* ptr = res.allocate_async(42, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_pinned_ptr(ptr);

    res.deallocate_async(ptr, 42, stream);
  }

  { // allocate_async / deallocate_async with alignment
    auto* ptr = res.allocate_async(42, 4, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

    stream.wait();
    ensure_pinned_ptr(ptr);

    res.deallocate_async(ptr, 42, 4, stream);
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
      try
      {
        auto* ptr = res.allocate_async(5, 42, stream);
        (void) ptr;
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
        auto* ptr = res.allocate_async(5, 1337, stream);
        (void) ptr;
      }
      catch (std::invalid_argument&)
      {
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
struct derived_pinned_resource : cudax::pinned_memory_resource
{
  using cudax::pinned_memory_resource::pinned_memory_resource;
};
static_assert(cuda::mr::resource<derived_pinned_resource>, "");

TEST_CASE("pinned_memory_resource comparison", "[memory_resource]")
{
  pinned_resource first{};
  { // comparison against a plain pinned_memory_resource
    pinned_resource second{};
    CHECK((first == second));
    CHECK(!(first != second));
  }

  { // comparison against a plain pinned_memory_resource with a different pool
    pinned_resource second{cudaMemAttachHost};
    CHECK((first != second));
    CHECK(!(first == second));
  }

  { // comparison against a pinned_memory_resource wrapped inside a resource_ref<device_accessible>
    pinned_resource second{};
    cuda::mr::resource_ref<cudax::device_accessible> const second_ref{second};
    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a pinned_memory_resource wrapped inside a async_resource_ref
    pinned_resource second{};
    // cuda::mr::async_resource_ref<cudax::device_accessible> second_ref{second};
    cudax::async_resource_ref<cudax::device_accessible> second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different pinned_resource through resource_ref
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

  { // comparison against a different pinned_resource through resource_ref
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
