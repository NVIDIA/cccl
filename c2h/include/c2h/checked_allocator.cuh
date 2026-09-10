// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/detail/__config>

#include <thrust/device_allocator.h>
#include <thrust/mr/new.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/pointer.h>

#include <cstddef>
#include <new>

#include <cuda_runtime_api.h>

#include <c2h/detail/checked_memory.cuh>

namespace c2h
{
using checked_cuda_memory_resource = THRUST_NS_QUALIFIER::system::cuda::detail::
  cuda_memory_resource<detail::checked_cuda_malloc, cudaFree, THRUST_NS_QUALIFIER::cuda::pointer<void>>;

template <typename T>
class checked_cuda_allocator
    : public THRUST_NS_QUALIFIER::mr::
        stateless_resource_allocator<T, THRUST_NS_QUALIFIER::device_ptr_memory_resource<checked_cuda_memory_resource>>
{
  using base = THRUST_NS_QUALIFIER::mr::
    stateless_resource_allocator<T, THRUST_NS_QUALIFIER::device_ptr_memory_resource<checked_cuda_memory_resource>>;

public:
  template <typename U>
  struct rebind
  {
    using other = checked_cuda_allocator<U>;
  };

  checked_cuda_allocator() = default;

  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator& other)
      : base(other)
  {}

  template <typename U>
  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator<U>& other)
      : base(other)
  {}

  checked_cuda_allocator& operator=(const checked_cuda_allocator&) = default;

  ~checked_cuda_allocator() = default;
};

struct checked_host_memory_resource final : public THRUST_NS_QUALIFIER::mr::new_delete_resource_base
{
  [[nodiscard]] _CCCL_HOST_API void*
  do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) final
  {
    // Some systems with integrated host/device memory have issues with allocating more memory
    // than is available. Check the amount of free memory before attempting to allocate on
    // integrated systems.
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
      throw std::bad_alloc{};
    }

    // Validate allocation-size arithmetic before delegating to new_delete_resource_base.
    const std::size_t allocation_size = detail::checked_host_allocation_size(bytes, alignment);

    if (detail::is_integrated_device(device))
    {
      const auto status = detail::check_free_device_memory(allocation_size);
      if (status != cudaSuccess)
      {
        throw std::bad_alloc{};
      }
    }

    return this->new_delete_resource_base::do_allocate(bytes, alignment);
  }
};

template <typename T>
using checked_host_allocator = THRUST_NS_QUALIFIER::mr::stateless_resource_allocator<T, checked_host_memory_resource>;
} // namespace c2h
