//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
#define CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H

#include <cuda/memory_resource>
#include <cuda/std/type_traits>

#include <cstdint>
#include <unordered_map>

#include <catch2/catch.hpp>

struct other_property
{};

//! @brief Simple wrapper around a memory resource that caches previous allocations
template <class Resource>
class caching_resource
{
private:
  Resource resource_{};
  std::unordered_multimap<std::size_t, void*> available_allocations_{};
  std::unordered_multimap<std::size_t, void*> used_allocations_{};

public:
  void* allocate(std::size_t size, std::size_t alignment)
  {
    if (available_allocations_.count(size) == 0)
    {
      return used_allocations_.emplace(size, resource_.allocate(size, alignment))->second;
    }
    else
    {
      return used_allocations_.insert(available_allocations_.extract(size))->second;
    }
  }
  void deallocate(void* ptr, std::size_t size, std::size_t)
  {
    CHECK(used_allocations_.count(size) != 0);
    auto range = used_allocations_.equal_range(size);
    for (auto curr = range.first; curr != range.second; ++curr)
    {
      if (curr->second == ptr)
      {
        available_allocations_.insert(used_allocations_.extract(curr));
        return;
      }
    }
    CHECK(false);
  }

  caching_resource() = default;

  ~caching_resource()
  {
    CHECK(used_allocations_.empty());
    for (const auto& elem : available_allocations_)
    {
      resource_.deallocate(elem.second, elem.first, alignof(int));
    }
  }

  bool operator==(const caching_resource& other) const
  {
    return resource_ == other.resource_;
  }
  bool operator!=(const caching_resource& other) const
  {
    return resource_ != other.resource_;
  }

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::has_property<Resource, Property>)
  friend void get_property(const caching_resource&, Property) noexcept {}

  // just add another property
  friend void get_property(const caching_resource&, other_property) noexcept {}
};

//! @brief Simple wrapper around a memory resource to ensure that it compares differently and we can test those code
//! paths
template <class... Properties>
struct memory_resource_wrapper
{
  cuda::mr::resource_ref<Properties...> ref_;

  void* allocate(std::size_t size, std::size_t alignment)
  {
    return ref_.allocate(size, alignment);
  }
  void deallocate(void* ptr, std::size_t size, std::size_t alignment)
  {
    ref_.deallocate(ptr, size, alignment);
  }

  bool operator==(const memory_resource_wrapper&) const
  {
    return true;
  }
  bool operator!=(const memory_resource_wrapper&) const
  {
    return false;
  }

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::std::__is_included_in<Property, Properties...>)
  friend void get_property(const memory_resource_wrapper&, Property) noexcept {}

  friend void get_property(const memory_resource_wrapper&, other_property) noexcept {}
};

//! @brief Memory resource that allocates host accessible memory via operator new
template <class T>
struct host_memory_resource
{
  void* allocate(std::size_t size, std::size_t)
  {
    return new T[size];
  }
  void deallocate(void* ptr, std::size_t, std::size_t)
  {
    delete[] reinterpret_cast<T*>(ptr);
  }

  bool operator==(const host_memory_resource&) const
  {
    return true;
  }
  bool operator!=(const host_memory_resource&) const
  {
    return false;
  }

  friend void get_property(const host_memory_resource&, cuda::mr::host_accessible) {}

  // just add another property
  friend void get_property(const host_memory_resource&, other_property) noexcept {}
};
static_assert(cuda::mr::resource<host_memory_resource<int>>, "");
static_assert(cuda::mr::resource_with<host_memory_resource<int>, cuda::mr::host_accessible>, "");

#endif // CUDAX_TEST_CONTAINER_VECTOR_TEST_RESOURCES_H
