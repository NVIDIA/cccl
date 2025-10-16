//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PTR_RANGES_OVERLAP_H
#define _CUDA___MEMORY_PTR_RANGES_OVERLAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/ptr_in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/addressof.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::forward_iterator<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool
ranges_overlap(_Tp __lhs_begin, _Tp __lhs_end, _Tp __rhs_begin, _Tp __rhs_end) noexcept
{
  if constexpr (::cuda::std::contiguous_iterator<_Tp>)
  {
    const auto __ptr_lhs_begin = ::cuda::std::addressof(__lhs_begin);
    const auto __ptr_lhs_end   = ::cuda::std::addressof(__lhs_end);
    const auto __ptr_rhs_begin = ::cuda::std::addressof(__rhs_begin);
    const auto __ptr_rhs_end   = ::cuda::std::addressof(__rhs_end);
    return ::cuda::ptr_in_range(__ptr_lhs_begin, __ptr_rhs_begin, __ptr_rhs_end)
        || ::cuda::ptr_in_range(__ptr_rhs_begin, __ptr_lhs_begin, __ptr_lhs_end);
  }
  else
  {
    return __lhs_begin < __rhs_end && __rhs_begin < __lhs_end;
  }

  _CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_PTR_RANGES_OVERLAP_H
