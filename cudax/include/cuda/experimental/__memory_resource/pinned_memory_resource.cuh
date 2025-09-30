//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H

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

#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/detail/libcxx/include/stdexcept>

#include <cuda/experimental/__memory_resource/memory_resource_base.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c pinned_memory_resource class provides a memory resource that allocates pinned memory.
namespace cuda::experimental
{

#if _CCCL_CUDACC_AT_LEAST(12, 6)

[[nodiscard]] static ::cudaMemPool_t __get_default_host_pinned_pool()
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  static ::cudaMemPool_t __default_pool = []() {
    ::cudaMemPool_t __pool = ::cuda::__driver::__getDefaultMemPool(
      ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0}, ::CU_MEM_ALLOCATION_TYPE_PINNED);
    // TODO should we be more careful with setting access from all devices? Maybe only if it was not set for any device?
    ::cuda::experimental::__mempool_set_access(__pool, ::cuda::devices, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    return __pool;
  }();

  return __default_pool;
#  else // _CCCL_CTK_BELOW(13, 0)
  static pinned_memory_resource __default_pool(0);
  return __default_pool.get();
#  endif // _CCCL_CTK_BELOW(13, 0)
}

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``pinned_memory_resource`` allocates pinned memory using `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__ for allocation/deallocation. A
//! ``pinned_memory_resource`` is a thin wrapper around a \c cudaMemPool_t with the location type set to \c
//! cudaMemLocationTypeHost or \c cudaMemLocationTypeHostNuma.
//!
//! .. warning::
//!
//!    ``pinned_memory_resource`` does not own the pool and it is the responsibility of the user to ensure that the
//!    lifetime of the pool exceeds the lifetime of the ``pinned_memory_resource``.
//!
//! @endrst
class pinned_memory_resource : public __memory_resource_base
{
public:
  //! @brief Default constructs the pinned_memory_resource using the default \c cudaMemPool_t for host pinned memory.
  //! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
  _CCCL_HOST_API pinned_memory_resource()
      : __memory_resource_base(::cuda::experimental::__get_default_host_pinned_pool())
  {}

  //! @brief  Constructs the pinned_memory_resource from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit pinned_memory_resource(::cudaMemPool_t __pool) noexcept
      : __memory_resource_base(__pool)
  {}

  //! @brief Enables the \c device_accessible property
  _CCCL_HOST_API friend constexpr void get_property(pinned_memory_resource const&, device_accessible) noexcept {}
  //! @brief Enables the \c host_accessible property
  _CCCL_HOST_API friend constexpr void get_property(pinned_memory_resource const&, host_accessible) noexcept {}

  using default_queries = properties_list<device_accessible, host_accessible>;
};

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``pinned_memory_pool`` allocates pinned memory using `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__ for allocation/deallocation.
//! When constructed it creates an underlying \c cudaMemPool_t with the location type set to \c cudaMemLocationTypeHost
//! or \c cudaMemLocationTypeHostNuma and owns it.
//!
//! @endrst
struct pinned_memory_pool : pinned_memory_resource
{
  using reference_type = pinned_memory_resource;

#  if _CCCL_CTK_AT_LEAST(13, 0)
  //! @brief Constructs a \c pinned_memory_pool with optional properties.
  //! Properties include the initial pool size and the release threshold. If the pool size grows beyond the release
  //! threshold, unused memory held by the pool will be released at the next synchronization event.

  //! @note Memory from this pool is accessible from all devices right away, which differs from the default behavior of
  //! pinned memory pools where memory is not accessible from devices until `cudaMemPoolSetAccess` is called.
  //!
  //! @param __properties Optional, additional properties of the pool to be created.
  _CCCL_HOST_API pinned_memory_pool(memory_pool_properties __properties = {})
      : pinned_memory_resource(__create_cuda_mempool(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0}, ::CU_MEM_ALLOCATION_TYPE_PINNED))
  {
    enable_access_from(cuda::devices);
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

  //! @brief Constructs a \c pinned_memory_pool with the specified NUMA node id and optional properties.
  //! Properties include the initial pool size and the release threshold. If the pool size grows beyond the release
  //! threshold, unused memory held by the pool will be released at the next synchronization event.
  //!
  //! @note Memory from this pool is accessible from all devices right away, which differs from the default behavior of
  //! pinned memory pools where memory is not accessible from devices until `cudaMemPoolSetAccess` is called.
  //!
  //! @param __numa_id The NUMA node id of the NUMA node the pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  _CCCL_HOST_API pinned_memory_pool(int __numa_id, memory_pool_properties __properties = {})
      : pinned_memory_resource(__create_cuda_mempool(
          __properties, ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST_NUMA, __numa_id}, ::CU_MEM_ALLOCATION_TYPE_PINNED))
  {
    enable_access_from(cuda::devices);
  }

  ~pinned_memory_pool() noexcept
  {
    ::cuda::__driver::__mempoolDestroy(__pool_);
  }

  _CCCL_HOST_API static pinned_memory_pool from_native_handle(::cudaMemPool_t __pool) noexcept
  {
    return pinned_memory_pool(__pool);
  }

  pinned_memory_pool(const pinned_memory_pool&)            = delete;
  pinned_memory_pool& operator=(const pinned_memory_pool&) = delete;

private:
  pinned_memory_pool(::cudaMemPool_t __pool) noexcept
      : pinned_memory_resource(__pool)
  {}
};

static_assert(::cuda::mr::resource_with<pinned_memory_resource, device_accessible>, "");
static_assert(::cuda::mr::resource_with<pinned_memory_resource, host_accessible>, "");

static_assert(::cuda::mr::resource_with<pinned_memory_pool, device_accessible>, "");
static_assert(::cuda::mr::resource_with<pinned_memory_pool, host_accessible>, "");

#endif // _CCCL_CUDACC_AT_LEAST(12, 6)

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H
