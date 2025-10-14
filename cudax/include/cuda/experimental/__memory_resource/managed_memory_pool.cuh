//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MANAGED_MEMORY_POOL_CUH
#define _CUDAX__MEMORY_RESOURCE_MANAGED_MEMORY_POOL_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(13, 0)

#  if _CCCL_CUDA_COMPILER(CLANG)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // _CCCL_CUDA_COMPILER(CLANG)

#  include <cuda/experimental/__memory_resource/memory_pool_base.cuh>
#  include <cuda/experimental/__stream/stream.cuh>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c managed_memory_pool class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental
{

class managed_memory_resource;

[[nodiscard]] static ::cudaMemPool_t __get_default_managed_pool()
{
  return ::cuda::__driver::__getDefaultMemPool(
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_NONE, 0}, ::CU_MEM_ALLOCATION_TYPE_MANAGED);
}

//! @brief \c managed_memory_pool is an owning wrapper around a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
//!
//! It handles creation and destruction of the underlying pool utilizing the provided \c memory_pool_properties.
class managed_memory_pool : public __memory_pool_base
{
private:
  //! @brief Constructs a \c managed_memory_pool from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  _CCCL_HOST_API explicit managed_memory_pool(__memory_pool_base::__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __memory_pool_base(__memory_pool_base::__from_handle_t{}, __handle)
  {}

public:
  //! @brief Constructs a \c managed_memory_pool with optional properties.
  //! Properties include the initial pool size and the release threshold. If the pool size grows beyond the release
  //! threshold, unused memory held by the pool will be released at the next synchronization event.
  //! @param __properties Optional, additional properties of the pool to be created.
  _CCCL_HOST_API explicit managed_memory_pool(memory_pool_properties __properties = {})
      : __memory_pool_base(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_NONE, 0}, ::CU_MEM_ALLOCATION_TYPE_MANAGED)
  {}

  // TODO add a constructor that accepts memory location one a type for it is added

  //! @brief Disables construction from a plain `cudaMemPool_t`. We want to ensure clean ownership semantics.
  managed_memory_pool(::cudaMemPool_t) = delete;

  managed_memory_pool(managed_memory_pool const&)            = delete;
  managed_memory_pool(managed_memory_pool&&)                 = delete;
  managed_memory_pool& operator=(managed_memory_pool const&) = delete;
  managed_memory_pool& operator=(managed_memory_pool&&)      = delete;

  //! @brief Construct an `pinned_memory_pool` object from a native `cudaMemPool_t` handle.
  //!
  //! @param __handle The native handle
  //!
  //! @return The constructed `pinned_memory_pool` object
  //!
  //! @note The constructed `pinned_memory_pool` object takes ownership of the native handle.
  [[nodiscard]] static managed_memory_pool from_native_handle(::cudaMemPool_t __handle) noexcept
  {
    return managed_memory_pool(__memory_pool_base::__from_handle_t{}, __handle);
  }

  // Disallow construction from an `int`, e.g., `0`.
  static managed_memory_pool from_native_handle(int) = delete;

  // Disallow construction from `nullptr`.
  static managed_memory_pool from_native_handle(::cuda::std::nullptr_t) = delete;

  using resource_type = managed_memory_resource;
};

} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(13, 0)

#endif // _CUDAX__MEMORY_RESOURCE_MANAGED_MEMORY_POOL_CUH
