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
#include <cuda/experimental/__memory_resource/pinned_memory_pool.cuh>

#include <cuda/std/__cccl/prologue.h>

// Trigger a rebuild of the file

//! @file
//! The \c pinned_memory_resource class provides a memory resource that allocates pinned memory.
namespace cuda::experimental
{

#if _CCCL_CUDACC_AT_LEAST(12, 6)

//! @rst
//! .. _cudax-memory-resource-async:
//!
//! Stream ordered memory resource
//! ------------------------------
//!
//! ``pinned_memory_resource`` uses `cudaMallocFromPoolAsync / cudaFreeAsync
//! <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`__ for allocation/deallocation. A
//! ``pinned_memory_resource`` is a thin wrapper around a \c cudaMemPool_t.
//!
//! .. warning::
//!
//!    ``pinned_memory_resource`` does not own the pool and it is the responsibility of the user to ensure that the
//!    lifetime of the pool exceeds the lifetime of the ``pinned_memory_resource``.
//!
//! @endrst
class pinned_memory_resource : public __memory_resource_base
{
private:
  //! @brief  Returns the default ``cudaMemPool_t`` for host pinned memory.
  //! @throws cuda_error if retrieving the default ``cudaMemPool_t`` fails.
  //! @returns The default memory pool for host pinned memory.
  [[nodiscard]] static ::cudaMemPool_t __get_default_sysmem_pool()
  {
    static pinned_memory_pool __default_pool{};
    return __default_pool.get();
  }

public:
  //! @brief Default constructs the pinned_memory_resource using the default \c cudaMemPool_t for host pinned memory.
  //! @throws cuda_error if retrieving the default \c cudaMemPool_t fails.
  pinned_memory_resource()
      : __memory_resource_base(__get_default_sysmem_pool())
  {}

  //! @brief  Constructs the pinned_memory_resource from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  explicit pinned_memory_resource(::cudaMemPool_t __pool) noexcept
      : __memory_resource_base(__pool)
  {}

  //! @brief  Constructs the pinned_memory_resource from a \c pinned_memory_pool by calling get().
  //! @param __pool The \c pinned_memory_pool used to allocate memory.
  explicit pinned_memory_resource(pinned_memory_pool& __pool) noexcept
      : __memory_resource_base(__pool.get())
  {}

  //! @brief Enables the \c device_accessible property
  friend constexpr void get_property(pinned_memory_resource const&, device_accessible) noexcept {}
  //! @brief Enables the \c host_accessible property
  friend constexpr void get_property(pinned_memory_resource const&, host_accessible) noexcept {}

  using default_queries = properties_list<device_accessible, host_accessible>;
};

static_assert(::cuda::mr::resource_with<pinned_memory_resource, device_accessible>, "");
static_assert(::cuda::mr::resource_with<pinned_memory_resource, host_accessible>, "");

#endif // _CCCL_CUDACC_AT_LEAST(12, 6)

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H
