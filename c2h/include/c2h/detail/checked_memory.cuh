// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__device/attributes.h>
#include <cuda/__memory/is_valid_alignment.h>
#include <cuda/__memory_resource/properties.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <new>

#include <cuda_runtime_api.h>

#include <c2h/detail/current_device.cuh>
#include <c2h/detail/env.cuh>

namespace c2h::detail
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
  static const std::size_t result = get_env_as_integer<std::size_t>("C2H_DEVICE_MEMORY_LIMIT");
  return result;
}

inline bool get_debug_checked_allocs()
{
  static const bool result = get_env_as_integer<long long>("C2H_DEBUG_CHECKED_ALLOC_FAILURES") != 0;
  return result;
}

inline cudaError_t get_device_memory(memory_info& info)
{
  static const std::size_t device_memory_limit = get_device_memory_limit();

  const cudaError_t status = cudaMemGetInfo(&info.free, &info.total);
  if (status != cudaSuccess)
  {
    return status;
  }

  if (device_memory_limit > 0)
  {
    const std::size_t limited_total     = (std::min) (info.total, device_memory_limit);
    const std::size_t unavailable_bytes = info.total - limited_total;
    info.free                           = info.free > unavailable_bytes ? info.free - unavailable_bytes : 0;
    info.total                          = limited_total;
    info.override                       = true;
  }

  return cudaSuccess;
}

inline cudaError_t check_free_device_memory(std::size_t bytes)
{
  memory_info info;
  const cudaError_t status = get_device_memory(info);
  if (status != cudaSuccess)
  {
    return status;
  }

  // Avoid allocating all available memory, i.e., if info.free < bytes + padding
  constexpr std::size_t padding = 16 * 1024 * 1024; // 16 MiB
  if (info.free < bytes || info.free - bytes < padding)
  {
    if (get_debug_checked_allocs())
    {
      const double total_GiB     = static_cast<double>(info.total) / (1024 * 1024 * 1024);
      const double free_GiB      = static_cast<double>(info.free) / (1024 * 1024 * 1024);
      const double requested_GiB = static_cast<double>(bytes) / (1024 * 1024 * 1024);
      const double padded_GiB    = requested_GiB + static_cast<double>(padding) / (1024 * 1024 * 1024);

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

// Returns whether alignment is valid and guaranteed to be satisfied by cudaMalloc.
[[nodiscard]] inline bool cuda_malloc_satisfies_alignment(std::size_t alignment) noexcept
{
  return ::cuda::__is_valid_alignment(alignment) && alignment <= ::cuda::mr::default_cuda_malloc_alignment;
}

[[nodiscard]] inline void* checked_device_allocate(int device, std::size_t bytes, std::size_t alignment)
{
  if (!cuda_malloc_satisfies_alignment(alignment))
  {
    throw std::bad_alloc{};
  }

  if (bytes == 0)
  {
    return nullptr;
  }

  assert_current_device(device);

  void* ptr                = nullptr;
  const cudaError_t status = checked_cuda_malloc(&ptr, bytes);
  if (status != cudaSuccess)
  {
    (void) cudaGetLastError();
    throw std::bad_alloc{};
  }

  return ptr;
}

inline void checked_device_deallocate(int device, void* ptr) noexcept
{
  if (ptr == nullptr)
  {
    return;
  }

  assert_current_device(device);
  (void) cudaFree(ptr);
}

[[nodiscard]] inline std::size_t checked_host_allocation_size(std::size_t bytes, std::size_t alignment)
{
  if (!::cuda::__is_valid_alignment(alignment))
  {
    throw std::bad_alloc{};
  }
  return bytes;
}

[[nodiscard]] inline bool is_integrated_device(int device)
{
  assert_current_device(device);

  // C2H test sessions target a single device, so the first queried device is the only device this function will see.
  static const bool result = [device] {
    int integrated{};
    if (cudaDeviceGetAttribute(&integrated, ::cuda::device_attributes::integrated, device) != cudaSuccess)
    {
      throw std::bad_alloc{};
    }
    return integrated != 0;
  }();
  return result;
}

[[nodiscard]] inline void* checked_host_allocate(int device, std::size_t bytes, std::size_t alignment)
{
  const std::size_t allocation_size = checked_host_allocation_size(bytes, alignment);

  if (bytes == 0)
  {
    return nullptr;
  }

  if (is_integrated_device(device))
  {
    const auto status = check_free_device_memory(allocation_size);
    if (status != cudaSuccess)
    {
      throw std::bad_alloc{};
    }
  }

  return ::operator new(bytes, std::align_val_t(alignment));
}

inline void checked_host_deallocate(void* ptr, [[maybe_unused]] std::size_t bytes, std::size_t alignment) noexcept
{
  if (ptr == nullptr)
  {
    return;
  }

#if __cpp_sized_deallocation >= 201309L
  ::operator delete(ptr, bytes, std::align_val_t(alignment));
#else // ^^^ __cpp_sized_deallocation >= 201309L ^^^ / vvv __cpp_sized_deallocation < 201309L vvv
  ::operator delete(ptr, std::align_val_t(alignment));
#endif // ^^^ __cpp_sized_deallocation < 201309L ^^^
}
} // namespace c2h::detail
