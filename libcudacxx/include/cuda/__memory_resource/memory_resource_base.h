//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_H
#define _CUDA___MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/get_memory_resource.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_MR

//! @brief CRTP base class for memory resources.
//!
//! When a resource inherits from `memory_resource_base<Derived>`, it provides a
//! `query(get_memory_resource_t)` method that returns a const reference to itself.
//! This enables the resource to be discovered inside a composed
//! `cuda::std::execution::env` by the `get_memory_resource` customization point.
template <class _Derived>
struct memory_resource_base
{
  [[nodiscard]] _CCCL_API constexpr const _Derived& query(const __get_memory_resource_t&) const noexcept
  {
    return static_cast<const _Derived&>(*this);
  }
};

_CCCL_END_NAMESPACE_CUDA_MR

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_RESOURCE_MEMORY_RESOURCE_BASE_H
