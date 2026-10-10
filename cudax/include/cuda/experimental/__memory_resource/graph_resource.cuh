//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_GRAPH_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_GRAPH_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/cstddef>

// Forward declaration to avoid circular include
namespace cuda::experimental
{
struct path_builder;
} // namespace cuda::experimental

namespace cuda::experimental
{
//! @brief The \c graph_resource concept verifies that a type provides graph-node-based
//! allocation and deallocation, plus stream-based deallocation for memory that outlives
//! the graph.
//!
//! A graph_resource must support:
//!   - ``allocate(path_builder& pb, size_t bytes, size_t alignment)`` → ``void*``
//!   - ``deallocate(path_builder& pb, void* ptr, size_t bytes, size_t alignment)``
//!   - ``deallocate(stream_ref stream, void* ptr, size_t bytes, size_t alignment)``
//!   - ``T() == T()``
//!   - ``T() != T()``
template <class _Resource>
_CCCL_CONCEPT graph_resource = _CCCL_REQUIRES_EXPR(
  (_Resource),
  _Resource& __res,
  path_builder& __pb,
  void* __ptr,
  size_t __bytes,
  size_t __alignment,
  ::cuda::stream_ref __stream)(
  _Same_as(void*) __res.allocate(__pb, __bytes, __alignment),
  _Same_as(void) __res.deallocate(__pb, __ptr, __bytes, __alignment),
  _Same_as(void) __res.deallocate(__stream, __ptr, __bytes, __alignment),
  requires(::cuda::std::equality_comparable<_Resource>));
} // namespace cuda::experimental

#endif // _CUDAX__MEMORY_RESOURCE_GRAPH_RESOURCE_CUH
