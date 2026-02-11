//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
//! @file
//! @brief Execution property and helpers for specifying allocation alignment.
//!
//! Provides the \c allocation_alignment execution property: when present in an
//! execution environment passed to container constructors (e.g. \c buffer),
//! it specifies the alignment to use for the allocation. The value must be a
//! power of two and not less than the element type's alignment.
//!
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_ALLOCATION_ALIGNMENT_H
#define _CUDA___MEMORY_RESOURCE_ALLOCATION_ALIGNMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__exception/throw_error.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Execution property type for querying allocation alignment from an environment.
struct allocation_alignment_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, allocation_alignment_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept -> ::cuda::std::size_t
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

//! @brief Execution property object: when bound in an environment (e.g. via \c execution::prop),
//! specifies the alignment to use for allocations. Used by \c buffer and related containers.
_CCCL_GLOBAL_CONSTANT auto allocation_alignment = allocation_alignment_t{};

//! @brief Returns true if \p __alignment is a power of two and not less than \p __min_alignment.
_CCCL_HOST_DEVICE inline constexpr bool
__is_valid_allocation_alignment(::cuda::std::size_t __alignment, ::cuda::std::size_t __min_alignment) noexcept
{
  return __alignment >= __min_alignment && __alignment != 0 && (__alignment & (__alignment - 1)) == 0;
}

//! @brief Throws std::invalid_argument if \p __alignment is not a valid allocation alignment
//! (power of two and at least \p __min_alignment).
_CCCL_HOST inline void
__validate_allocation_alignment(::cuda::std::size_t __alignment, ::cuda::std::size_t __min_alignment)
{
  if (!__is_valid_allocation_alignment(__alignment, __min_alignment))
  {
    ::cuda::std::__throw_invalid_argument(
      "Invalid allocation alignment: must be a power of two and at least the "
      "type's alignment.");
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_RESOURCE_ALLOCATION_ALIGNMENT_H
