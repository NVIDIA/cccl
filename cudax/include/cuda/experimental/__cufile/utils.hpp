//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <stdexcept>

#include <cuda_runtime.h>

namespace cuda::experimental::cufile
{

namespace utils
{

//! Check if a pointer is GPU memory
inline bool is_gpu_memory(const void* ptr)
{
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  return (err == cudaSuccess) && (attrs.type == cudaMemoryTypeDevice);
}

//! Get device ID for a GPU memory pointer
inline int get_device_id(const void* ptr)
{
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess)
  {
    throw ::std::runtime_error("Failed to get pointer attributes");
  }
  return attrs.device;
}

//! Get optimal alignment for cuFile operations
[[nodiscard]] constexpr size_t get_optimal_alignment()
{
  return 4096; // 4KB alignment for most file systems
}

//! Check if a pointer is suitable for cuFile operations
inline bool is_cufile_compatible(const void* ptr)
{
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess)
  {
    return false;
  }

  // cuFile works with device memory and registered host memory
  return (attrs.type == cudaMemoryTypeDevice) || (attrs.type == cudaMemoryTypeHost);
}

} // namespace utils

} // namespace cuda::experimental::cufile
