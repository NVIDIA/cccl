//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_PINNED_MEMORY_POOL_H
#define _CUDA___MEMORY_RESOURCE_PINNED_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__device/all_devices.h>
#  include <cuda/__memory_pool/memory_pool_base.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/throw_error.h>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c pinned_memory_resource class provides a memory resource that
//! allocates pinned memory.
_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CTK_AT_LEAST(12, 6)

static ::cudaMemPool_t __get_default_host_pinned_pool();

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered host pinned memory pool
//! ------------------------------
//!
//! ``pinned_memory_pool_ref`` allocates pinned memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. A
//! ``pinned_memory_pool_ref`` is a thin wrapper around a \c cudaMemPool_t with
//! the location type set to \c cudaMemLocationTypeHost or \c
//! cudaMemLocationTypeHostNuma.
//!
//! .. warning::
//!
//!    ``pinned_memory_pool_ref`` does not own the pool and it is the
//!    responsibility of the user to ensure that the lifetime of the pool
//!    exceeds the lifetime of the ``pinned_memory_pool_ref``.
//!
//! @endrst
class pinned_memory_pool_ref : public __memory_pool_base
{
public:
  //! @brief  Constructs the pinned_memory_pool_ref from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit pinned_memory_pool_ref(::cudaMemPool_t __pool) noexcept
      : __memory_pool_base(__pool)
  {}

  //! @brief Enables the \c device_accessible property
  _CCCL_HOST_API friend constexpr void
  get_property(pinned_memory_pool_ref const&, ::cuda::mr::device_accessible) noexcept
  {}
  //! @brief Enables the \c host_accessible property
  _CCCL_HOST_API friend constexpr void get_property(pinned_memory_pool_ref const&, ::cuda::mr::host_accessible) noexcept
  {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>;
};

//! @brief Returns the default pinned memory pool.
//! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
//! @returns The default pinned memory pool.
[[nodiscard]] inline pinned_memory_pool_ref pinned_default_memory_pool()
{
  return pinned_memory_pool_ref{::cuda::__get_default_host_pinned_pool()};
}

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``pinned_memory_pool`` allocates pinned memory using
//! `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__
//! for allocation/deallocation. When constructed it creates an underlying \c
//! cudaMemPool_t with the location type set to \c cudaMemLocationTypeHost or \c
//! cudaMemLocationTypeHostNuma and owns it.
//!
//! @endrst
struct pinned_memory_pool : pinned_memory_pool_ref
{
  using reference_type = pinned_memory_pool_ref;

#    if _CCCL_CTK_AT_LEAST(13, 0)
  //! @brief Constructs a \c pinned_memory_pool with optional properties.
  //! Properties include the initial pool size and the release threshold. If the
  //! pool size grows beyond the release threshold, unused memory held by the
  //! pool will be released at the next synchronization event.

  //! @note Memory from this pool is accessible from all devices right away,
  //! which differs from the default behavior of pinned memory pools where
  //! memory is not accessible from devices until `cudaMemPoolSetAccess` is
  //! called.
  //!
  //! @param __properties Optional, additional properties of the pool to be
  //! created.
  _CCCL_HOST_API pinned_memory_pool(memory_pool_properties __properties = {})
      : pinned_memory_pool_ref(__create_cuda_mempool(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0}, ::CU_MEM_ALLOCATION_TYPE_PINNED))
  {
    enable_access_from(cuda::devices);
  }
#    endif // _CCCL_CTK_AT_LEAST(13, 0)

  //! @brief Constructs a \c pinned_memory_pool with the specified NUMA node id
  //! and optional properties. Properties include the initial pool size and the
  //! release threshold. If the pool size grows beyond the release threshold,
  //! unused memory held by the pool will be released at the next
  //! synchronization event.
  //!
  //! @note Memory from this pool is accessible from all devices right away,
  //! which differs from the default behavior of pinned memory pools where
  //! memory is not accessible from devices until `cudaMemPoolSetAccess` is
  //! called.
  //!
  //! @param __numa_id The NUMA node id of the NUMA node the pool is constructed
  //! on.
  //! @param __pool_properties Optional, additional properties of the pool to be
  //! created.
  _CCCL_HOST_API pinned_memory_pool(int __numa_id, memory_pool_properties __properties = {})
      : pinned_memory_pool_ref(__create_cuda_mempool(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST_NUMA, __numa_id}, ::CU_MEM_ALLOCATION_TYPE_PINNED))
  {
    enable_access_from(cuda::devices);
  }

  ~pinned_memory_pool() noexcept
  {
    if (__pool_ != nullptr)
    {
      ::cuda::__driver::__mempoolDestroy(__pool_);
    }
  }

  _CCCL_HOST_API static pinned_memory_pool from_native_handle(::cudaMemPool_t __pool) noexcept
  {
    return pinned_memory_pool(__pool);
  }

  //! @brief Returns a \c pinned_memory_pool_ref for this \c pinned_memory_pool.
  //! The result is the same as if this object was cast to a \c pinned_memory_pool_ref.
  _CCCL_HOST_API pinned_memory_pool_ref as_ref() noexcept
  {
    return pinned_memory_pool_ref(__pool_);
  }

  pinned_memory_pool(const pinned_memory_pool&)            = delete;
  pinned_memory_pool& operator=(const pinned_memory_pool&) = delete;

private:
  pinned_memory_pool(::cudaMemPool_t __pool) noexcept
      : pinned_memory_pool_ref(__pool)
  {}
};

static_assert(::cuda::mr::resource_with<pinned_memory_pool_ref, ::cuda::mr::device_accessible>, "");
static_assert(::cuda::mr::resource_with<pinned_memory_pool_ref, ::cuda::mr::host_accessible>, "");

static_assert(::cuda::mr::resource_with<pinned_memory_pool, ::cuda::mr::device_accessible>, "");
static_assert(::cuda::mr::resource_with<pinned_memory_pool, ::cuda::mr::host_accessible>, "");

[[nodiscard]] static ::cudaMemPool_t __get_default_host_pinned_pool()
{
#    if _CCCL_CTK_AT_LEAST(13, 0)
  static ::cudaMemPool_t __default_pool = []() {
    ::cudaMemPool_t __pool = ::cuda::__get_default_memory_pool(
      ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0}, ::CU_MEM_ALLOCATION_TYPE_PINNED);
    // TODO should we be more careful with setting access from all devices?
    // Maybe only if it was not set for any device?
    ::cuda::__mempool_set_access(__pool, ::cuda::devices, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    return __pool;
  }();

#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  static ::cudaMemPool_t __default_pool = []() {
    cuda::pinned_memory_pool __pool(0);
    return __pool.release();
  }();
#    endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
  return __default_pool;
}

#  endif // _CCCL_CTK_AT_LEAST(12, 6)

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_PINNED_MEMORY_POOL_H
