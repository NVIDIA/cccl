//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

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
  friend void get_property(const resource&, cuda::mr::pinned_memory) noexcept
  {}
};
static_assert(cuda::mr::resource<resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::resource_with<resource<AccessibilityType::Host>, cuda::mr::pinned_memory>, "");
static_assert(cuda::mr::resource<resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::resource_with<resource<AccessibilityType::Device>, cuda::mr::pinned_memory>, "");

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
static_assert(!cuda::mr::async_resource_with<async_resource<AccessibilityType::Host>, cuda::mr::pinned_memory>, "");
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::async_resource_with<async_resource<AccessibilityType::Device>, cuda::mr::pinned_memory>, "");

void test()
{
  cuda::mr::cuda_pinned_memory_resource first{};
  { // comparison against a plain cuda_pinned_memory_resource
    cuda::mr::cuda_pinned_memory_resource second{cudaHostAllocDefault};
    assert(first == second);
    assert(!(first != second));
  }

  { // comparison against a plain cuda_pinned_memory_resource with a different flag set
    cuda::mr::cuda_pinned_memory_resource second{cudaHostAllocPortable};
    assert(!(first == second));
    assert((first != second));
  }

  { // comparison against a cuda_pinned_memory_resource wrapped inside a resource_ref<cuda::mr::pinned_memory>
    cuda::mr::cuda_pinned_memory_resource second{};
    cuda::mr::resource_ref<cuda::mr::pinned_memory> second_ref{second};
    assert(first == second_ref);
    assert(!(first != second_ref));
    assert(second_ref == first);
    assert(!(second_ref != first));
  }

  { // comparison against a cuda_pinned_memory_resource wrapped inside a resource_ref<>
    cuda::mr::cuda_pinned_memory_resource second{};
    cuda::mr::resource_ref<> second_ref{second};
    assert(first == second_ref);
    assert(!(first != second_ref));
    assert(second_ref == first);
    assert(!(second_ref != first));
  }

  { // comparison against a different resource through resource_ref
    resource<AccessibilityType::Host> host_resource{};
    resource<AccessibilityType::Device> device_resource{};
    assert(!(first == host_resource));
    assert(first != host_resource);
    assert(!(first == device_resource));
    assert(first != device_resource);

    assert(!(host_resource == first));
    assert(host_resource != first);
    assert(!(device_resource == first));
    assert(device_resource != first);
  }

  { // comparison against a different resource through resource_ref
    async_resource<AccessibilityType::Host> host_async_resource{};
    async_resource<AccessibilityType::Device> device_async_resource{};
    assert(!(first == host_async_resource));
    assert(first != host_async_resource);
    assert(!(first == device_async_resource));
    assert(first != device_async_resource);

    assert(!(host_async_resource == first));
    assert(host_async_resource != first);
    assert(!(device_async_resource == first));
    assert(device_async_resource != first);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
