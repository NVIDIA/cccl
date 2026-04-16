//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_QUERIES_INDEX_H
#define _CUDA___HIERARCHY_QUERIES_INDEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/hierarchy_query_result.h>
#  include <cuda/__hierarchy/queries/extents.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>

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
struct __index_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Unit, _Level>, "_Level must be reachable from _Unit");

    using _NextLevel       = typename _Unit::__next_native_level;
    const auto __curr_exts = __extents_query_native<_Unit, _NextLevel>::template __call<_Tp>();
    const auto __next_idx  = __index_query_native<_NextLevel, _Level>::template __call<_Tp>();
    const auto __curr_idx  = __index_query_native<_Unit, _NextLevel>::template __call<_Tp>();

    hierarchy_query_result<_Tp> __ret{};
    for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
    {
      __ret[__i] = __curr_idx[__i] + ((__i < __curr_exts.rank()) ? __curr_exts.extent(__i) : 1) * __next_idx[__i];
    }
    return __ret;
  }
};

template <>
struct __index_query_native<thread_level, warp_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    // todo(dabayer): Is it worth using cuda::ptx::get_sreg_laneid() here? Doesn't it prevent some other optimizations
    // due to using inline ptx?
    return {static_cast<_Tp>(::cuda::ptx::get_sreg_laneid()), 0, 0};
  }
};

template <>
struct __index_query_native<thread_level, block_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    return {static_cast<_Tp>(threadIdx.x), static_cast<_Tp>(threadIdx.y), static_cast<_Tp>(threadIdx.z)};
  }
};

template <>
struct __index_query_native<warp_level, block_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    const auto __thread_rank = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    return {static_cast<_Tp>(__thread_rank / 32), 0, 0};
  }
};

template <>
struct __index_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    ::dim3 __idx{0u, 0u, 0u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__idx = ::__clusterRelativeBlockIdx();))
    return {static_cast<_Tp>(__idx.x), static_cast<_Tp>(__idx.y), static_cast<_Tp>(__idx.z)};
  }
};

template <>
struct __index_query_native<block_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    return {static_cast<_Tp>(blockIdx.x), static_cast<_Tp>(blockIdx.y), static_cast<_Tp>(blockIdx.z)};
  }
};

template <>
struct __index_query_native<cluster_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call() noexcept
  {
    ::dim3 __idx{blockIdx};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__idx = ::__clusterIdx();))
    return {static_cast<_Tp>(__idx.x), static_cast<_Tp>(__idx.y), static_cast<_Tp>(__idx.z)};
  }
};

// hierarchy queries

template <class _Tp, class _Unit, class _NextLevel, class _Level, class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API hierarchy_query_result<_Tp> __index_query_generic(const _Hierarchy& __hier) noexcept
{
  if constexpr (::cuda::std::is_same_v<_Level, _NextLevel>)
  {
    using _CurrExts = decltype(__extents_query<_Unit, _NextLevel>::template __call<_Tp>(__hier));
    auto __curr_idx = __index_query_native<_Unit, _NextLevel>::template __call<_Tp>();
    for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
    {
      if (__i >= _CurrExts::rank() || _CurrExts::static_extent(__i) == 1)
      {
        __curr_idx[__i] = 0;
      }
    }
    return __curr_idx;
  }
  else
  {
    const auto __curr_exts = __extents_query<_Unit, _NextLevel>::template __call<_Tp>(__hier);
    const auto __next_idx  = __index_query<_NextLevel, _Level>::template __call<_Tp>(__hier);
    const auto __curr_idx  = __index_query_native<_Unit, _NextLevel>::template __call<_Tp>();

    hierarchy_query_result<_Tp> __ret{};
    for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
    {
      __ret[__i] = __curr_idx[__i] + ((__i < __curr_exts.rank()) ? __curr_exts.extent(__i) : 1) * __next_idx[__i];
    }
    return __ret;
  }
}

template <class _Unit, class _Level>
struct __index_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy& __hier) noexcept
  {
    static_assert(__has_bottom_unit_or_level_v<_Unit, _Hierarchy> || __is_native_hierarchy_level_v<_Unit>,
                  "_Hierarchy doesn't contain _Unit");
    static_assert(_Hierarchy::template has_level<_Level>() || __is_native_hierarchy_level_v<_Level>,
                  "_Hierarchy doesn't contain _Level");

    using _NextLevel = __next_hierarchy_level_t<_Unit, _Hierarchy>;
    return ::cuda::__index_query_generic<_Tp, _Unit, _NextLevel, _Level>(__hier);
  }
};

template <>
struct __index_query<thread_level, warp_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy& __hier) noexcept
  {
    return __index_query_native<thread_level, warp_level>::template __call<_Tp>();
  }
};

template <class _Level>
struct __index_query<warp_level, _Level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy& __hier) noexcept
  {
    const auto __block_exts = __extents_query<thread_level, block_level>::template __call<unsigned>(__hier);
    const auto __thread_idx = __index_query<thread_level, block_level>::template __call<unsigned>(__hier);
    const auto __thread_rank =
      (__thread_idx.z * __block_exts.extent(1) + __thread_idx.y) * __block_exts.extent(0) + __thread_idx.x;
    const auto __warp_rank = __thread_rank / 32;

    if constexpr (::cuda::std::is_same_v<_Level, block_level>)
    {
      return {static_cast<_Tp>(__warp_rank), 0, 0};
    }
    else
    {
      const auto __thread_count = __block_exts.extent(0) * __block_exts.extent(1) * __block_exts.extent(2);
      const auto __warp_count   = ::cuda::ceil_div(__thread_count, 32);
      const auto __next_idx     = __index_query<block_level, _Level>::template __call<_Tp>(__hier);
      return {static_cast<_Tp>(__next_idx.x * __warp_count + __warp_rank), __next_idx.y, __next_idx.z};
    }
  }
};

template <>
struct __index_query<block_level, cluster_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy&) noexcept
  {
    return __index_query_native<block_level, cluster_level>::template __call<_Tp>();
  }
};

template <>
struct __index_query<block_level, grid_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy&) noexcept
  {
    return __index_query_native<block_level, grid_level>::template __call<_Tp>();
  }
};

template <>
struct __index_query<cluster_level, grid_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> __call(const _Hierarchy& __hier) noexcept
  {
    return __index_query_native<cluster_level, grid_level>::template __call<_Tp>();
  }
};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_QUERIES_INDEX_H
