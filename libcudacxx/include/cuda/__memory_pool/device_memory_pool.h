//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_DEVICE_MEMORY_POOL_H
#define _CUDA___MEMORY_RESOURCE_DEVICE_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__memory_pool/memory_pool_base.h>
#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/std/__concepts/concept_macros.h>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c device_memory_pool class provides an asynchronous memory resource
//! that allocates device memory in stream order.
_CCCL_BEGIN_NAMESPACE_CUDA

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory pool
//! ------------------------------
//!
//! ``device_memory_pool_ref`` allocates device memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. A
//! ``device_memory_pool_ref`` is a thin wrapper around a \c cudaMemPool_t with
//! the location type set to \c cudaMemLocationTypeDevice.
//!
//! .. warning::
//!
//!    ``device_memory_pool_ref`` does not own the pool and it is the
//!    responsibility of the user to ensure that the lifetime of the pool
//!    exceeds the lifetime of the ``device_memory_pool_ref``.
//!
//! @endrst
class device_memory_pool_ref : public __memory_pool_base
{
public:
  //! @brief  Constructs the device_memory_pool_ref from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit device_memory_pool_ref(::cudaMemPool_t __pool) noexcept
      : __memory_pool_base(__pool)
  {}

  device_memory_pool_ref(int)                    = delete;
  device_memory_pool_ref(::cuda::std::nullptr_t) = delete;

  //! @brief Enables the \c device_accessible property for \c
  //! device_memory_pool_ref.
  //! @relates device_memory_pool_ref
  _CCCL_HOST_API friend constexpr void
  get_property(device_memory_pool_ref const&, ::cuda::mr::device_accessible) noexcept
  {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;
};

//! @brief  Returns the default ``cudaMemPool_t`` from the specified device.
//! @throws cuda_error if retrieving the default ``cudaMemPool_t`` fails.
//! @returns The default memory pool of the specified device.
[[nodiscard]] inline device_memory_pool_ref device_default_memory_pool(::cuda::device_ref __device)
{
  static ::cudaMemPool_t __pool = ::cuda::__get_default_memory_pool(
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __device.get()}, ::CU_MEM_ALLOCATION_TYPE_PINNED);
  return device_memory_pool_ref(__pool);
}

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``device_memory_pool`` allocates device memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. A When constructed it creates an underlying \c
//! cudaMemPool_t with the location type set to \c cudaMemLocationTypeDevice and
//! owns it.
//!
//! @endrst
struct device_memory_pool : device_memory_pool_ref
{
  using reference_type = device_memory_pool_ref;

  //! @brief Constructs a \c device_memory_pool with the optionally specified
  //! initial pool size and release threshold. If the pool size grows beyond the
  //! release threshold, unused memory held by the pool will be released at the
  //! next synchronization event.
  //! @throws cuda_error if the CUDA version does not support
  //! ``cudaMallocAsync``.
  //! @param __device_id The device id of the device the stream pool is
  //! constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be
  //! created.
  _CCCL_HOST_API device_memory_pool(::cuda::device_ref __device_id, memory_pool_properties __properties = {})
      : device_memory_pool_ref(__create_cuda_mempool(
          __properties,
          ::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __device_id.get()},
          ::CU_MEM_ALLOCATION_TYPE_PINNED))
  {}

  ~device_memory_pool() noexcept
  {
    if (__pool_ != nullptr)
    {
      ::cuda::__driver::__mempoolDestroy(__pool_);
    }
  }

  _CCCL_HOST_API static device_memory_pool from_native_handle(::cudaMemPool_t __pool) noexcept
  {
    return device_memory_pool(__pool);
  }

  //! @brief Returns a \c device_memory_pool_ref for this \c device_memory_pool.
  //! The result is the same as if this object was cast to a \c device_memory_pool_ref.
  [[nodiscard]] _CCCL_HOST_API device_memory_pool_ref as_ref() noexcept
  {
    return device_memory_pool_ref(__pool_);
  }

  device_memory_pool(const device_memory_pool&)            = delete;
  device_memory_pool& operator=(const device_memory_pool&) = delete;

private:
  device_memory_pool(::cudaMemPool_t __pool) noexcept
      : device_memory_pool_ref(__pool)
  {}
};

static_assert(::cuda::mr::synchronous_resource_with<device_memory_pool_ref, ::cuda::mr::device_accessible>, "");

static_assert(::cuda::mr::resource_with<device_memory_pool, ::cuda::mr::device_accessible>, "");

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_DEVICE_MEMORY_POOL_H
