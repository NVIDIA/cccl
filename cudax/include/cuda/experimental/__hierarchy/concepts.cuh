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

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// todo:
//   - add check that level_type is actually a valid level type
//   - check that at least level_type{}.count(__g) and level_type{}.rank(__g) are valid?
template <class _Tp>
_CCCL_CONCEPT hierarchy_group = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __g)(
  typename(typename _Tp::level_type), //
  __g.sync() //
);
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_CONCEPTS_CUH
