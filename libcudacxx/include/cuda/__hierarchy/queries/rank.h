//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_QUERIES_RANK_H
#define _CUDA___HIERARCHY_QUERIES_RANK_H

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
#  include <cuda/__hierarchy/hierarchy_query_result.h>
#  include <cuda/__hierarchy/queries/count.h>
#  include <cuda/__hierarchy/queries/extents.h>
#  include <cuda/__hierarchy/queries/index.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__cstddef/types.h>

#  if _CCCL_CUDA_COMPILATION()
#    include <cuda/__ptx/instructions/get_sreg.h>
#  endif // _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CUDA_COMPILATION()

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#    if _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
#    endif // _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)

// native hierarchy queries

template <class _Unit, class _Level>
struct __rank_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    using _NextLevel = typename _Unit::__next_native_level;

    const auto __curr_exts = __extents_query_native<_Unit, _NextLevel>::template __call<_Tp>();
    const auto __curr_idx  = __index_query_native<_Unit, _NextLevel>::template __call<_Tp>();

    _Tp __ret = 0;
    if constexpr (!::cuda::std::is_same_v<_Level, _NextLevel>)
    {
      __ret = __rank_query_native<_NextLevel, _Level>::template __call<_Tp>()
            * __count_query_native<_Unit, _NextLevel>::template __call<_Tp>();
    }

    for (::cuda::std::size_t __i = __curr_exts.rank(); __i > 0; --__i)
    {
      _Tp __inc = __curr_idx[__i - 1];
      for (::cuda::std::size_t __j = __i - 1; __j > 0; --__j)
      {
        __inc *= __curr_exts.extent(__j - 1);
      }
      __ret += __inc;
    }
    return __ret;
  }
};

template <>
struct __rank_query_native<thread_level, warp_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return static_cast<_Tp>(::cuda::ptx::get_sreg_laneid());
  }
};

template <>
struct __rank_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    unsigned __rank = 0;
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__rank = ::__clusterRelativeBlockRank();))
    return static_cast<_Tp>(__rank);
  }
};

template <>
struct __rank_query_native<block_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call() noexcept
  {
    return static_cast<_Tp>((static_cast<_Tp>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
  }
};

// hierarchy queries

template <class _Tp, class _Unit, class _NextLevel, class _Level, class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API _Tp __rank_query_generic(const _Hierarchy& __hier) noexcept
{
  const auto __curr_exts = __extents_query<_Unit, _NextLevel>::template __call<_Tp>(__hier);
  const auto __curr_idx  = __index_query<_Unit, _NextLevel>::template __call<_Tp>(__hier);

  _Tp __ret = 0;
  if constexpr (!::cuda::std::is_same_v<_Level, _NextLevel>)
  {
    __ret = __rank_query<_NextLevel, _Level>::template __call<_Tp>(__hier)
          * __count_query<_Unit, _NextLevel>::template __call<_Tp>(__hier);
  }

  for (::cuda::std::size_t __i = __curr_exts.rank(); __i > 0; --__i)
  {
    _Tp __inc = __curr_idx[__i - 1];
    for (::cuda::std::size_t __j = __i - 1; __j > 0; --__j)
    {
      __inc *= __curr_exts.extent(__j - 1);
    }
    __ret += __inc;
  }
  return __ret;
}

template <class _Unit, class _Level>
struct __rank_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    using _NextLevel = __next_hierarchy_level_t<_Unit, _Hierarchy>;
    return ::cuda::__rank_query_generic<_Tp, _Unit, _NextLevel, _Level>(__hier);
  }
};

template <>
struct __rank_query<thread_level, warp_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy&) noexcept
  {
    return __rank_query_native<thread_level, warp_level>::template __call<_Tp>();
  }
};

template <class _Level>
struct __rank_query<warp_level, _Level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    return ::cuda::__rank_query_generic<_Tp, warp_level, block_level, _Level>(__hier);
  }
};

template <>
struct __rank_query<block_level, cluster_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy&) noexcept
  {
    return __rank_query_native<block_level, cluster_level>::template __call<_Tp>();
  }
};

template <>
struct __rank_query<block_level, grid_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy&) noexcept
  {
    return __rank_query_native<block_level, grid_level>::template __call<_Tp>();
  }
};

template <>
struct __rank_query<cluster_level, grid_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp __call(const _Hierarchy& __hier) noexcept
  {
    return ::cuda::__rank_query_generic<_Tp, cluster_level, grid_level, grid_level>(__hier);
  }
};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_QUERIES_RANK_H
