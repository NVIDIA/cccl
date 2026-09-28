// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <cuda_runtime_api.h>

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
  static std::optional<std::string> override_str = get_env("C2H_DEVICE_MEMORY_LIMIT");
  static const std::size_t result =
    override_str ? static_cast<std::size_t>(std::strtoll(override_str->c_str(), nullptr, 10)) : 0;
  return result;
}

inline bool get_debug_checked_allocs()
{
  static std::optional<std::string> debug_checked_allocs = get_env("C2H_DEBUG_CHECKED_ALLOC_FAILURES");
  static const bool result = debug_checked_allocs && (std::strtol(debug_checked_allocs->c_str(), nullptr, 10) != 0);
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
} // namespace c2h::detail
