//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_QUERIES_COUNT_H
#define _CUDA___HIERARCHY_QUERIES_COUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/std/__cstddef/types.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// native hierarchy queries

#  if _CCCL_CUDA_COMPILATION()

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#    if _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
#    endif // _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)

template <class _Unit, class _Level>
struct __count_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    const auto __exts = __extents_query_native<_Unit, _Level>::template __call<_Tp>();

    _Tp __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }
};

template <>
struct __count_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    unsigned __count = 1;
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__count = ::__clusterSizeInBlocks();))
    return static_cast<_Tp>(__count);
  }
};

template <>
struct __count_query_native<block_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return static_cast<_Tp>(static_cast<_Tp>(gridDim.x) * gridDim.y * gridDim.z);
  }
};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION()

// hierarchy queries

template <class _Unit, class _Level>
struct __count_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr _Tp __call(const _Hierarchy& __hier) noexcept
  {
    const auto __exts = __extents_query<_Unit, _Level>::template __call<_Tp>(__hier);

    _Tp __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_QUERIES_COUNT_H
