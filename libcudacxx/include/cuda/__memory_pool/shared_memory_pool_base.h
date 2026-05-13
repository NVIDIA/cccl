//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_POOL_SHARED_MEMORY_POOL_BASE_H
#define _CUDA___MEMORY_POOL_SHARED_MEMORY_POOL_BASE_H

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
#  include <cuda/__memory_resource/memory_resource_base.h>
#  include <cuda/__memory_resource/shared_block_ptr.h>
#  include <cuda/__utility/no_init.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief RAII wrapper that destroys a ``cudaMemPool_t`` on destruction.
//! Used as the payload of a ``__shared_block_ptr`` to tie pool lifetime
//! to the shared reference count.
struct __pool_destroyer
{
  ::cudaMemPool_t __pool;

  _CCCL_HOST_API explicit __pool_destroyer(::cudaMemPool_t __p) noexcept
      : __pool(__p)
  {}

  _CCCL_HOST_API __pool_destroyer(__pool_destroyer&& __other) noexcept
      : __pool(::cuda::std::exchange(__other.__pool, nullptr))
  {}

  _CCCL_HOST_API ~__pool_destroyer() noexcept
  {
    if (__pool != nullptr)
    {
      _CCCL_ASSERT_CUDA_API(::cuda::__driver::__mempoolDestroyNoThrow, "Failed to destroy a memory pool", __pool);
    }
  }

  __pool_destroyer(const __pool_destroyer&)            = delete;
  __pool_destroyer& operator=(const __pool_destroyer&) = delete;
  __pool_destroyer& operator=(__pool_destroyer&&)      = delete;
};

//! @brief CRTP base for shared memory pool types.
//!
//! Inherits from ``__memory_pool_base`` so that all pool operations
//! (allocate, deallocate, trim_to, attribute, enable_access_from, …) are
//! available directly. A ``__shared_block_ptr<__pool_destroyer>`` ties the
//! pool handle lifetime to a shared reference count: copies share the pool
//! and the last owner destroys it.
//!
//! Derived types supply constructors, ``get_property`` friend overloads, and
//! ``default_queries``.
template <class _Derived>
class __shared_memory_pool_base
    : public __memory_pool_base
    , public ::cuda::mr::memory_resource_base<_Derived>
{
  ::cuda::mr::__shared_block_ptr<__pool_destroyer> __ref_;

protected:
  _CCCL_HOST_API explicit __shared_memory_pool_base(no_init_t) noexcept
      : __memory_pool_base(::cudaMemPool_t{})
  {}

  _CCCL_HOST_API explicit __shared_memory_pool_base(::cudaMemPool_t __pool)
      : __memory_pool_base(__pool)
      , __ref_(__pool)
  {}

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_API __shared_memory_pool_base(const __shared_memory_pool_base& __other) noexcept
      : __memory_pool_base(__other.__pool_)
      , __ref_(__other.__ref_)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_API __shared_memory_pool_base(__shared_memory_pool_base&& __other) noexcept
      : __memory_pool_base(::cuda::std::exchange(__other.__pool_, nullptr))
      , __ref_(::cuda::std::move(__other.__ref_))
  {}

  __shared_memory_pool_base& operator=(const __shared_memory_pool_base&) = default;
  __shared_memory_pool_base& operator=(__shared_memory_pool_base&&)      = default;

  //! @brief ``release()`` is deleted because ownership is shared.
  ::cudaMemPool_t release() = delete;
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_POOL_SHARED_MEMORY_POOL_BASE_H
