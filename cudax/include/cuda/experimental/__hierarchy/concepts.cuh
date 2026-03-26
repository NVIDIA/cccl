//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_CONCEPTS_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_CONCEPTS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/extents.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _Group>
_CCCL_CONCEPT group = _CCCL_REQUIRES_EXPR((_Group), _Group&& __g, const _Group&& __cg)(
  typename(typename _Group::unit_type),
  requires(__is_hierarchy_level_v<typename _Group::unit_type>),
  typename(typename _Group::level_type),
  requires(__is_hierarchy_level_v<typename _Group::level_type>),
  typename(typename _Group::hierarchy_type),
  requires(__is_hierarchy_v<typename _Group::hierarchy_type>),
  _Same_as(void) __g.sync(),
  _Same_as(void) __g.sync_aligned(),
  _Same_as(const typename _Group::hierarchy_type&) __cg.hierarchy()
  // todo: add __sub_unit_queryable and __super_unit_queryable
);
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_CONCEPTS_CUH
