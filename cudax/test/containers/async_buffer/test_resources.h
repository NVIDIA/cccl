//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
#define CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H

#include <cuda/memory_resource>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

#include <cuda/experimental/memory_resource.cuh>

#include <cstdint>
#include <unordered_map>

#include <testing.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct other_property
{};

// make the cudax resources have that property for tests
inline void get_property(const cuda::experimental::device_memory_resource&, other_property) {}
inline void get_property(const cuda::experimental::legacy_pinned_memory_resource&, other_property) {}
#if _CCCL_CUDACC_AT_LEAST(12, 6)
inline void get_property(const cuda::experimental::pinned_memory_resource&, other_property) {}
#endif

//! @brief Simple wrapper around a memory resource to ensure that it compares differently and we can test those code
//! paths
template <class... Properties>
struct memory_resource_wrapper
{
  // Not a resource_ref, because it can't be used to create any_resource (yet)
  // https://github.com/NVIDIA/cccl/issues/4166
  cudax::any_resource<Properties...> resource_;

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

#endif // CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
