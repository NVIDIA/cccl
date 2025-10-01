//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 0)

#  include <cuda/__memory_resource/properties.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/detail/libcxx/include/stdexcept>

#  include <cuda/experimental/__memory_resource/managed_memory_pool.cuh>
#  include <cuda/experimental/__memory_resource/memory_resource_base.cuh>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c managed_memory_resource class provides a memory resource that allocates managed memory.
namespace cuda::experimental
{

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``managed_memory_resource`` uses `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__ for allocation/deallocation. A
//! ``managed_memory_resource`` is a thin wrapper around a \c cudaMemPool_t.
//!
//! .. warning::
//!
//!    ``managed_memory_resource`` does not own the pool and it is the responsibility of the user to ensure that the
//!    lifetime of the pool exceeds the lifetime of the ``managed_memory_resource``.
//!
//! @endrst
class managed_memory_resource : public __memory_resource_base
{
public:
  //! @brief Default constructs the managed_memory_resource using the default \c cudaMemPool_t for host pinned memory.
  //! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
  _CCCL_HOST_API managed_memory_resource()
      : __memory_resource_base(::cuda::experimental::__get_default_managed_pool())
  {}

  //! @brief  Constructs the managed_memory_resource from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit managed_memory_resource(::cudaMemPool_t __pool) noexcept
      : __memory_resource_base(__pool)
  {}

  //! @brief  Constructs the managed_memory_resource from a \c managed_memory_pool by calling get().
  //! @param __pool The \c managed_memory_pool used to allocate memory.
  _CCCL_HOST_API explicit managed_memory_resource(managed_memory_pool& __pool) noexcept
      : __memory_resource_base(__pool.get())
  {}

  //! @brief Enables the \c device_accessible property
  _CCCL_HOST_API friend constexpr void get_property(managed_memory_resource const&, device_accessible) noexcept {}
  //! @brief Enables the \c host_accessible property
  _CCCL_HOST_API friend constexpr void get_property(managed_memory_resource const&, host_accessible) noexcept {}

  using default_queries = properties_list<device_accessible, host_accessible>;
};

static_assert(::cuda::mr::resource_with<managed_memory_resource, device_accessible>, "");
static_assert(::cuda::mr::resource_with<managed_memory_resource, host_accessible>, "");

} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 0)

#endif //_CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H
