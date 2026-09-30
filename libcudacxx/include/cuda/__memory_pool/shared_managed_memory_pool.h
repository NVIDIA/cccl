//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_POOL_SHARED_MANAGED_MEMORY_POOL_H
#define _CUDA___MEMORY_POOL_SHARED_MANAGED_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(13, 0)

#  include <cuda/__memory_pool/managed_memory_pool.h>
#  include <cuda/__memory_pool/shared_memory_pool_base.h>
#  include <cuda/__memory_resource/properties.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @rst
//! .. _libcudacxx-memory-pool-shared-managed:
//!
//! Shared managed memory pool
//! --------------------------
//!
//! ``shared_managed_memory_pool`` provides shared ownership of a managed memory
//! pool. It is copyable and each copy shares the same underlying
//! ``cudaMemPool_t`` via reference counting. This makes it usable with
//! ``any_resource`` and other contexts that require copyable resources.
//!
//! @endrst
class shared_managed_memory_pool : public __shared_memory_pool_base<shared_managed_memory_pool>
{
public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_API shared_managed_memory_pool( // NOLINT(modernize-use-equals-default)
    const shared_managed_memory_pool& __other) noexcept
      : __shared_memory_pool_base(__other)
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_API shared_managed_memory_pool( // NOLINT(modernize-use-equals-default)
    shared_managed_memory_pool&& __other) noexcept
      : __shared_memory_pool_base(::cuda::std::move(__other))
  {}
  shared_managed_memory_pool& operator=(const shared_managed_memory_pool&) = default;
  shared_managed_memory_pool& operator=(shared_managed_memory_pool&&)      = default;

  //! @brief Constructs an empty shared managed memory pool.
  _CCCL_HOST_API explicit shared_managed_memory_pool(no_init_t) noexcept
      : __shared_memory_pool_base(no_init)
  {}

  //! @brief Constructs a shared managed memory pool.
  //! @param __properties Optional pool creation properties.
  _CCCL_HOST_API explicit shared_managed_memory_pool(memory_pool_properties __properties = {})
      : __shared_memory_pool_base(managed_memory_pool(__properties).release())
  {}

  //! @brief Constructs a shared managed memory pool from an existing native handle.
  //! @param __pool The ``cudaMemPool_t`` to take shared ownership of.
  [[nodiscard]] _CCCL_HOST_API static shared_managed_memory_pool from_native_handle(::cudaMemPool_t __pool) noexcept
  {
    return shared_managed_memory_pool(__pool);
  }

  _CCCL_HOST_API friend constexpr void
  get_property(const shared_managed_memory_pool&, ::cuda::mr::device_accessible) noexcept
  {}
  _CCCL_HOST_API friend constexpr void
  get_property(const shared_managed_memory_pool&, ::cuda::mr::host_accessible) noexcept
  {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>;

private:
  _CCCL_HOST_API explicit shared_managed_memory_pool(::cudaMemPool_t __pool) noexcept
      : __shared_memory_pool_base(__pool)
  {}
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(13, 0)

#endif // _CUDA___MEMORY_POOL_SHARED_MANAGED_MEMORY_POOL_H
