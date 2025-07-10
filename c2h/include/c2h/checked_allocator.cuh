/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/device_allocator.h>
#include <thrust/mr/new.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/system/cuda/pointer.h>

#include <cstdlib>
#include <iostream>
#include <new>

#include <cuda_runtime_api.h>

namespace c2h
{
namespace detail
{

struct memory_info
{
  std::size_t free{};
  std::size_t total{};
  bool override{false};
};

// If the environment variable C2H_DEVICE_MEMORY_LIMIT is set, the total device memory
// will be limited to this number of bytes.
inline std::size_t get_device_memory_limit()
{
  static const char* override_str = std::getenv("C2H_DEVICE_MEMORY_LIMIT");
  static std::size_t result       = override_str ? static_cast<std::size_t>(std::atoll(override_str)) : 0;
  return result;
}

inline bool get_debug_checked_allocs()
{
  static const char* debug_checked_allocs = std::getenv("C2H_DEBUG_CHECKED_ALLOC_FAILURES");
  static bool result                      = debug_checked_allocs && (std::atoi(debug_checked_allocs) != 0);
  return result;
}

inline cudaError_t get_device_memory(memory_info& info)
{
  static std::size_t device_memory_limit = get_device_memory_limit();

  cudaError_t status = cudaMemGetInfo(&info.free, &info.total);
  if (status != cudaSuccess)
  {
    return status;
  }

  if (device_memory_limit > 0)
  {
    info.free  = (std::max) (std::size_t{0}, static_cast<std::size_t>(info.free - (info.total - device_memory_limit)));
    info.total = device_memory_limit;
    info.override = true;
  }

  return cudaSuccess;
}

inline cudaError_t check_free_device_memory(std::size_t bytes)
{
  memory_info info;
  cudaError_t status = get_device_memory(info);
  if (status != cudaSuccess)
  {
    return status;
  }

  // Avoid allocating all available memory:
  constexpr std::size_t padding = 16 * 1024 * 1024; // 16 MiB
  if (info.free < (bytes + padding))
  {
    if (get_debug_checked_allocs())
    {
      const double total_GiB     = static_cast<double>(info.total) / (1024 * 1024 * 1024);
      const double free_GiB      = static_cast<double>(info.free) / (1024 * 1024 * 1024);
      const double requested_GiB = static_cast<double>(bytes) / (1024 * 1024 * 1024);
      const double padded_GiB    = static_cast<double>(bytes + padding) / (1024 * 1024 * 1024);

      std::cerr << "Device memory allocation failed due to insufficient free device memory.\n";

      if (info.override)
      {
        std::cerr
          << "Available device memory has been limited (env var C2H_DEVICE_MEMORY_LIMIT=" << get_device_memory_limit()
          << ").\n";
      }

      std::cerr
        << "Total device mem:     " << total_GiB << " GiB\n" //
        << "Free device mem:      " << free_GiB << " GiB\n" //
        << "Requested device mem: " << requested_GiB << " GiB\n" //
        << "Padded device mem:    " << padded_GiB << " GiB\n";
    }

    return cudaErrorMemoryAllocation;
  }

  return cudaSuccess;
}

// Check available memory prior to calling cudaMalloc.
// This avoids hangups and slowdowns from allocating swap / non-device memory
// on some platforms, namely tegra.
inline cudaError_t checked_cuda_malloc(void** ptr, std::size_t bytes)
{
  auto status = check_free_device_memory(bytes);
  if (status != cudaSuccess)
  {
    return status;
  }

  return cudaMalloc(ptr, bytes);
}
} // namespace detail

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

  _CCCL_HOST_DEVICE checked_cuda_allocator() {}

  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator& other)
      : base(other)
  {}

  template <typename U>
  _CCCL_HOST_DEVICE checked_cuda_allocator(const checked_cuda_allocator<U>& other)
      : base(other)
  {}

  checked_cuda_allocator& operator=(const checked_cuda_allocator&) = default;

  _CCCL_HOST_DEVICE ~checked_cuda_allocator() {}
};

struct checked_host_memory_resource final : public THRUST_NS_QUALIFIER::mr::new_delete_resource_base
{
  void* do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) final
  {
    // Some systems with integrated host/device memory have issues with allocating more memory
    // than is available. Check the amount of free memory before attempting to allocate on
    // integrated systems.
    int device = 0;
    CubDebugExit(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CubDebugExit(cudaGetDeviceProperties(&prop, device));
    if (prop.integrated)
    {
      auto status = detail::check_free_device_memory(bytes + alignment + sizeof(std::size_t));
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
