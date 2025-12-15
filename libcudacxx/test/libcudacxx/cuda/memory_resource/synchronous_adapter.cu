//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/stream>

#include <testing.cuh>

#include "test_resource.cuh"

template <class Resource>
constexpr bool same_default_queries =
  cuda::std::is_same_v<typename cuda::mr::synchronous_resource_adapter<Resource>::default_queries,
                       typename Resource::default_queries>;

template <class Resource, class Property>
constexpr bool passed_property =
  cuda::mr::synchronous_resource_with<cuda::mr::synchronous_resource_adapter<Resource>, Property>
  == cuda::mr::synchronous_resource_with<Resource, Property>;

template <class Resource>
constexpr bool same_properties =
  passed_property<Resource, cuda::mr::device_accessible> && passed_property<Resource, cuda::mr::host_accessible>
  && passed_property<Resource, extra_property> && passed_property<Resource, get_data>;

C2H_CCCLRT_TEST("synchronous_resource_adapter", "[memory_resource]")
{
  cuda::stream stream{cuda::device_ref{0}};

  SECTION("Test wrapping a resource")
  {
    auto pool = cuda::device_default_memory_pool(cuda::device_ref{0});
    cuda::mr::synchronous_resource_adapter<cuda::device_memory_pool_ref> adapter{pool};
    auto* ptr = adapter.allocate(stream, 1024, 128);
    CCCLRT_CHECK(ptr != nullptr);
    CCCLRT_CHECK(pool.attribute(cuda::memory_pool_attributes::used_mem_current) > 0);
    adapter.deallocate(stream, ptr, 1024, 128);
    CCCLRT_CHECK(pool.attribute(cuda::memory_pool_attributes::used_mem_current) == 0);
  }
  SECTION("Test wrapping a synchronous resource")
  {
    cuda::mr::synchronous_resource_adapter<cuda::mr::legacy_pinned_memory_resource> adapter{
      cuda::mr::legacy_pinned_memory_resource{}};
    auto* ptr = adapter.allocate(stream, 1024, 128);
    CCCLRT_CHECK(ptr != nullptr);
    adapter.deallocate(stream, ptr, 1024, 128);
  }
  SECTION("test property passing through")
  {
#if _CCCL_CTK_AT_LEAST(12, 6)
    STATIC_CHECK(same_properties<cuda::pinned_memory_pool_ref>);
#endif // _CCCL_CTK_AT_LEAST(12, 6)
    STATIC_CHECK(same_properties<cuda::device_memory_pool_ref>);
    STATIC_CHECK(same_properties<cuda::mr::legacy_pinned_memory_resource>);
    STATIC_CHECK(same_properties<small_resource>);
  }
  SECTION("test default queries")
  {
#if _CCCL_CTK_AT_LEAST(12, 6)
    STATIC_CHECK(same_default_queries<cuda::pinned_memory_pool_ref>);
#endif // _CCCL_CTK_AT_LEAST(12, 6)
    STATIC_CHECK(same_default_queries<cuda::device_memory_pool_ref>);
    STATIC_CHECK(same_default_queries<cuda::mr::legacy_pinned_memory_resource>);
    STATIC_CHECK(same_default_queries<small_resource>);
  }
}
