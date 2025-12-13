//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_MANAGED_MEMORY_POOL_H
#define _CUDA___MEMORY_RESOURCE_MANAGED_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(13, 0)

#  include <cuda/__memory_pool/memory_pool_base.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/throw_error.h>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c managed_memory_resource class provides a memory resource that
//! allocates managed memory.
_CCCL_BEGIN_NAMESPACE_CUDA

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``managed_memory_pool_ref`` allocates managed memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. A
//! ``managed_memory_pool_ref`` is a thin wrapper around a \c cudaMemPool_t with
//! the allocation type set to \c cudaMemAllocationTypeManaged.
//!
//! .. warning::
//!
//!    ``managed_memory_pool_ref`` does not own the pool and it is the
//!    responsibility of the user to ensure that the lifetime of the pool
//!    exceeds the lifetime of the ``managed_memory_pool_ref``.
//!
//! @endrst
class managed_memory_pool_ref : public __memory_pool_base
{
public:
  //! @brief  Constructs the managed_memory_pool_ref from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit managed_memory_pool_ref(::cudaMemPool_t __pool) noexcept
      : __memory_pool_base(__pool)
  {}

  //! @brief Enables the \c device_accessible property
  _CCCL_HOST_API friend constexpr void
  get_property(managed_memory_pool_ref const&, ::cuda::mr::device_accessible) noexcept
  {}
  //! @brief Enables the \c host_accessible property
  _CCCL_HOST_API friend constexpr void get_property(managed_memory_pool_ref const&, ::cuda::mr::host_accessible) noexcept
  {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>;
};

//! @brief Returns the default managed memory pool.
//! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
//! @returns The default managed memory pool.
[[nodiscard]] inline managed_memory_pool_ref managed_default_memory_pool()
{
  static ::cudaMemPool_t __pool = ::cuda::__get_default_memory_pool(
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_NONE, 0}, ::CU_MEM_ALLOCATION_TYPE_MANAGED);
  return managed_memory_pool_ref(__pool);
}

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``managed_memory_pool`` allocates managed memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. A When constructed it creates an underlying \c
//! cudaMemPool_t with the allocation type set to \c
//! cudaMemAllocationTypeManaged and owns it.
//!
//! @endrst
struct managed_memory_pool : managed_memory_pool_ref
{
  using reference_type = managed_memory_pool_ref;

  //! @brief Constructs a \c managed_memory_pool with optional properties.
  //! Properties include the initial pool size and the release threshold. If the
  //! pool size grows beyond the release threshold, unused memory held by the
  //! pool will be released at the next synchronization event.
  //! @param __properties Optional, additional properties of the pool to be
  //! created.
  _CCCL_HOST_API managed_memory_pool(memory_pool_properties __properties = {})
      : managed_memory_pool_ref(__create_cuda_mempool(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_NONE, 0}, ::CU_MEM_ALLOCATION_TYPE_MANAGED))
  {}

  // TODO add a constructor that accepts memory location one a type for it is
  // added

  ~managed_memory_pool() noexcept
  {
    if (__pool_ != nullptr)
    {
      ::cuda::__driver::__mempoolDestroy(__pool_);
    }
  }

  _CCCL_HOST_API static managed_memory_pool from_native_handle(::cudaMemPool_t __pool) noexcept
  {
    return managed_memory_pool(__pool);
  }

  //! @brief Returns a \c managed_memory_pool_ref for this \c managed_memory_pool.
  //! The result is the same as if this object was cast to a \c managed_memory_pool_ref.
  [[nodiscard]] _CCCL_HOST_API managed_memory_pool_ref as_ref() noexcept
  {
    return managed_memory_pool_ref(__pool_);
  }

  managed_memory_pool(const managed_memory_pool&)            = delete;
  managed_memory_pool& operator=(const managed_memory_pool&) = delete;

private:
  managed_memory_pool(::cudaMemPool_t __pool) noexcept
      : managed_memory_pool_ref(__pool)
  {}
};

static_assert(::cuda::mr::resource_with<managed_memory_pool_ref, ::cuda::mr::device_accessible>, "");
static_assert(::cuda::mr::resource_with<managed_memory_pool_ref, ::cuda::mr::host_accessible>, "");

static_assert(::cuda::mr::resource_with<managed_memory_pool, ::cuda::mr::device_accessible>, "");
static_assert(::cuda::mr::resource_with<managed_memory_pool, ::cuda::mr::host_accessible>, "");

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(13, 0)

#endif //_CUDA___MEMORY_RESOURCE_MANAGED_MEMORY_POOL_H
