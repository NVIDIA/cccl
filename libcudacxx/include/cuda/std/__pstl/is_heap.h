//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_IS_HEAP_H
#define _CUDA_STD___PSTL_IS_HEAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/is_heap.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__pstl/is_heap_until.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _RandomAccessIterator, class _Compare = less<>)
_CCCL_REQUIRES(__has_random_access_traversal<_RandomAccessIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API bool
is_heap(const _Policy& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp = {})
{
  return ::cuda::std::is_heap_until(__policy, __first, __last, ::cuda::std::move(__comp)) == __last;
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_IS_HEAP_H
