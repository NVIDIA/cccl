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

#include <testing.cuh>
#include <utility.cuh>

#if _CCCL_CTK_AT_LEAST(13, 0)
#  define TEST_TYPES cuda::legacy_managed_memory_resource, cuda::managed_memory_pool_ref
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
#  define TEST_TYPES cuda::legacy_managed_memory_resource
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

template <typename Resource>
void resource_static_asserts()
{
  static_assert(!cuda::std::is_trivial<Resource>::value, "");
  static_assert(!cuda::std::is_trivially_default_constructible<Resource>::value, "");
  static_assert(cuda::std::is_trivially_copy_constructible<Resource>::value, "");
  static_assert(cuda::std::is_trivially_move_constructible<Resource>::value, "");
  static_assert(cuda::std::is_trivially_copy_assignable<Resource>::value, "");
  static_assert(cuda::std::is_trivially_move_assignable<Resource>::value, "");
  static_assert(cuda::std::is_trivially_destructible<Resource>::value, "");
  static_assert(!cuda::std::is_empty<Resource>::value, "");
}

template void resource_static_asserts<cuda::legacy_managed_memory_resource>();
#if _CCCL_CTK_AT_LEAST(13, 0)
template void resource_static_asserts<cuda::managed_memory_pool_ref>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <class Resource>
Resource get_resource()
{
#if _CCCL_CTK_AT_LEAST(13, 0)
  if constexpr (cuda::std::is_same_v<Resource, cuda::managed_memory_pool_ref>)
  {
    return cuda::managed_default_memory_pool();
  }
  else
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  {
    return Resource{};
  }
}

static void ensure_managed_ptr(void* ptr)
{
  CHECK(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  CHECK(status == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeManaged);
}

C2H_CCCLRT_TEST_LIST("managed_memory_resource construction", "[memory_resource]", TEST_TYPES)
{
  using managed_resource = TestType;
  SECTION("Default construction")
  {
    if constexpr (cuda::std::is_same_v<managed_resource, cuda::legacy_managed_memory_resource>)
    {
      STATIC_REQUIRE(cuda::std::is_default_constructible_v<managed_resource>);
    }
  }

#if _CCCL_CTK_BELOW(13, 0)
  SECTION("Construct with flag")
  {
    managed_resource defaulted{};
    managed_resource with_flag{cudaMemAttachHost};
    CHECK(defaulted != with_flag);
  }
#endif // _CCCL_CTK_BELOW(13, 0)
}

C2H_CCCLRT_TEST_LIST("managed_memory_resource allocation", "[memory_resource]", TEST_TYPES)
{
  using managed_resource = TestType;
  managed_resource res   = get_resource<managed_resource>();
  cuda::stream stream{cuda::device_ref{0}};

  { // allocate_sync / deallocate_sync
    auto* ptr = res.allocate_sync(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_managed_ptr(ptr);

    res.deallocate_sync(ptr, 42);
  }

  { // allocate_sync / deallocate_sync with alignment
    auto* ptr = res.allocate_sync(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_managed_ptr(ptr);

    res.deallocate_sync(ptr, 42, 4);
  }

  if constexpr (cuda::mr::resource<managed_resource>)
  {
    { // allocate / deallocate
      auto* ptr = res.allocate(stream, 42);
      static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

      stream.sync();
      ensure_managed_ptr(ptr);

      res.deallocate(stream, ptr, 42);
    }

    { // allocate / deallocate with alignment
      auto* ptr = res.allocate(stream, 42, 4);
      static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");

      stream.sync();
      ensure_managed_ptr(ptr);

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
struct derived_managed_resource : cuda::legacy_managed_memory_resource
{
  using cuda::legacy_managed_memory_resource::legacy_managed_memory_resource;
};
static_assert(cuda::mr::synchronous_resource<derived_managed_resource>, "");

C2H_CCCLRT_TEST_LIST("managed_memory_resource comparison", "[memory_resource]", TEST_TYPES)
{
  using managed_resource = TestType;
  managed_resource first = get_resource<managed_resource>();
  { // comparison against a plain managed_memory_resource
    managed_resource second = get_resource<managed_resource>();
    CHECK((first == second));
    CHECK(!(first != second));
  }

  if constexpr (cuda::std::is_same_v<managed_resource, cuda::legacy_managed_memory_resource>)
  { // comparison against a plain legacy_managed_memory_resource with a different flags
    managed_resource second = cuda::legacy_managed_memory_resource{cudaMemAttachHost};
    CHECK((first != second));
    CHECK(!(first == second));
  }
#if _CCCL_CTK_AT_LEAST(13, 0)
  else
  {
    // comparison against a managed_memory_pool_ref with a different pool
    cuda::managed_memory_pool second{};
    CHECK((first != second));
    CHECK(!(first == second));
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  { // comparison against a managed_memory_resource wrapped inside a synchronous_resource_ref<device_accessible>
    managed_resource second = get_resource<managed_resource>();
    cuda::mr::synchronous_resource_ref<::cuda::mr::device_accessible> second_ref{second};
    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different managed_resource through synchronous_resource_ref
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

  if constexpr (cuda::mr::resource<managed_resource>)
  { // comparison against a managed_memory_resource wrapped inside a resource_ref
    managed_resource second = get_resource<managed_resource>();
    cuda::mr::resource_ref<::cuda::mr::device_accessible> second_ref{second};

    CHECK((first == second_ref));
    CHECK(!(first != second_ref));
    CHECK((second_ref == first));
    CHECK(!(second_ref != first));
  }

  { // comparison against a different managed_resource through synchronous_resource_ref
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
#if _CCCL_CTK_AT_LEAST(13, 0)
// async deallocate_sync test removed in this suite; covered elsewhere
#endif // _CCCL_CTK_AT_LEAST(13, 0)
