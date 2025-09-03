//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL_CUH
#define _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__new_>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__memory_resource/memory_pool_base.cuh>
#include <cuda/experimental/__stream/stream.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c device_memory_pool class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental
{

//! @brief \c device_memory_pool is an owning wrapper around a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
//!
//! It handles creation and destruction of the underlying pool utilizing the provided \c memory_pool_properties.
class device_memory_pool : public __memory_pool_base
{
  //! @brief Constructs a \c device_memory_pool from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  explicit device_memory_pool(__memory_pool_base::__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __memory_pool_base(__memory_pool_base::__from_handle_t{}, __handle)
  {}

public:
  //! @brief Constructs a \c device_memory_pool with the optionally specified initial pool size and release threshold.
  //! If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next
  //! synchronization event.
  //! @throws cuda_error if the CUDA version does not support ``cudaMallocAsync``.
  //! @param __device_id The device id of the device the stream pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  explicit device_memory_pool(const ::cuda::device_ref __device_id, memory_pool_properties __properties = {})
      : __memory_pool_base(__memory_location_type::__device, __properties, __device_id.get())
  {}

  //! @brief Disables construction from a plain `cudaMemPool_t`. We want to ensure clean ownership semantics.
  device_memory_pool(::cudaMemPool_t) = delete;

  device_memory_pool(device_memory_pool const&)            = delete;
  device_memory_pool(device_memory_pool&&)                 = delete;
  device_memory_pool& operator=(device_memory_pool const&) = delete;
  device_memory_pool& operator=(device_memory_pool&&)      = delete;

  //! @brief Construct an `device_memory_pool` object from a native `cudaMemPool_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return The constructed `device_memory_pool` object
  //!
  //! @note The constructed `device_memory_pool` object takes ownership of the native handle.
  [[nodiscard]] static device_memory_pool from_native_handle(::cudaMemPool_t __handle) noexcept
  {
    return device_memory_pool(__memory_pool_base::__from_handle_t{}, __handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static device_memory_pool from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static device_memory_pool from_native_handle(::cuda::std::nullptr_t) = delete;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_DEVICE_MEMORY_POOL_CUH
