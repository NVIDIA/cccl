//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
#define CUDA_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H

#include <cuda/__stream/stream_ref.h>
#include <cuda/memory_resource>
#include <cuda/std/type_traits>

#include <cstdint>
#include <unordered_map>

#include <testing.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct other_property
{};

// make the cudax resources have that property for tests
inline void get_property(const cuda::device_memory_pool_ref&, other_property) {}
inline void get_property(const cuda::mr::legacy_pinned_memory_resource&, other_property) {}
#if _CCCL_CTK_AT_LEAST(12, 6)
inline void get_property(const cuda::pinned_memory_pool_ref&, other_property) {}
#endif // _CCCL_CTK_AT_LEAST(12, 6)

//! @brief Simple wrapper around a memory resource to ensure that it compares
//! differently and we can test those code paths
template <class... Properties>
struct memory_resource_wrapper
{
  // Not a resource_ref, because it can't be used to create any_resource (yet)
  // https://github.com/NVIDIA/cccl/issues/4166
  cuda::mr::any_resource<Properties...> resource_;

  void* allocate_sync(std::size_t size, std::size_t alignment)
  {
    return resource_.allocate_sync(size, alignment);
  }
  void deallocate_sync(void* ptr, std::size_t size, std::size_t alignment)
  {
    resource_.deallocate_sync(ptr, size, alignment);
  }
  void* allocate(cuda::stream_ref stream, std::size_t size, std::size_t alignment)
  {
    return resource_.allocate(stream, size, alignment);
  }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t size, std::size_t alignment)
  {
    resource_.deallocate(stream, ptr, size, alignment);
  }

  bool operator==(const memory_resource_wrapper&) const
  {
    return true;
  }
  bool operator!=(const memory_resource_wrapper&) const
  {
    return false;
  }

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES(cuda::std::__is_included_in_v<Property, Properties...>)
  friend void get_property(const memory_resource_wrapper&, Property) noexcept {}

  friend void get_property(const memory_resource_wrapper&, other_property) noexcept {}
};

// Adapter that offsets the pointer by the alignment to enable testing that the resource was passed the correct
// alignment.
template <typename Resource>
struct offset_by_alignment_resource
    : ::cuda::mr::__copy_default_queries<Resource>
    , ::cuda::forward_property<offset_by_alignment_resource<Resource>, Resource>
{
  Resource resource_;

  offset_by_alignment_resource(const Resource& resource) noexcept
      : resource_(resource)
  {}
  offset_by_alignment_resource(Resource&& resource) noexcept
      : resource_(cuda::std::move(resource))
  {}

  void* offset_by_alignment(void* ptr, std::size_t alignment)
  {
    return reinterpret_cast<char*>(ptr) + alignment;
  }

  void* remove_alignment_offset(void* ptr, std::size_t alignment)
  {
    return reinterpret_cast<char*>(ptr) - alignment;
  }

  void* allocate_sync(std::size_t size, std::size_t alignment)
  {
    return offset_by_alignment(resource_.allocate_sync(size + alignment, alignment), alignment);
  }
  void deallocate_sync(void* ptr, std::size_t size, std::size_t alignment)
  {
    resource_.deallocate_sync(remove_alignment_offset(ptr, alignment), size + alignment, alignment);
  }
  void* allocate(cuda::stream_ref stream, std::size_t size, std::size_t alignment)
  {
    return offset_by_alignment(resource_.allocate(stream, size + alignment, alignment), alignment);
  }

  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t size, std::size_t alignment)
  {
    resource_.deallocate(stream, remove_alignment_offset(ptr, alignment), size + alignment, alignment);
  }

  bool operator==(const offset_by_alignment_resource&) const
  {
    return true;
  }
  bool operator!=(const offset_by_alignment_resource&) const
  {
    return false;
  }

  Resource& upstream_resource() noexcept
  {
    return resource_;
  }
  const Resource& upstream_resource() const noexcept
  {
    return resource_;
  }
};

#endif // CUDA_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
