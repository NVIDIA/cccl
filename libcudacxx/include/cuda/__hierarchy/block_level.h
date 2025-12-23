//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_BLOCK_LEVEL_H
#define _CUDA___HIERARCHY_BLOCK_LEVEL_H

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
#  include <cuda/__hierarchy/native_hierarchy_level_base.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct block_level : __native_hierarchy_level_base<block_level>
{
  using product_type  = unsigned;
  using allowed_above = allowed_levels<grid_level, cluster_level>;
  using allowed_below = allowed_levels<thread_level>;

  using __next_native_level = cluster_level;

  using __base_type = __native_hierarchy_level_base<block_level>;
  using __base_type::count_as;
  using __base_type::extents_as;

#  if _CCCL_CUDA_COMPILATION()
  using __base_type::index_as;
  using __base_type::rank_as;

  // interactions with cluster level

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> extents_as(const cluster_level&) noexcept
  {
    ::dim3 __dims{1u, 1u, 1u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__dims = ::__clusterDim();))
    return ::cuda::std::dims<3, _Tp>{static_cast<_Tp>(__dims.x), static_cast<_Tp>(__dims.y), static_cast<_Tp>(__dims.z)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp count_as(const cluster_level&) noexcept
  {
    unsigned __count = 1;
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__count = ::__clusterSizeInBlocks();))
    return static_cast<_Tp>(__count);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> index_as(const cluster_level&) noexcept
  {
    ::dim3 __idx{0u, 0u, 0u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__idx = ::__clusterRelativeBlockIdx();))
    return {static_cast<_Tp>(__idx.x), static_cast<_Tp>(__idx.y), static_cast<_Tp>(__idx.z)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp rank_as(const cluster_level&) noexcept
  {
    unsigned __rank = 0;
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__rank = ::__clusterRelativeBlockRank();))
    return static_cast<_Tp>(__rank);
  }

  // interactions with grid level

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> extents_as(const grid_level&) noexcept
  {
    return ::cuda::std::dims<3, _Tp>{
      static_cast<_Tp>(gridDim.x), static_cast<_Tp>(gridDim.y), static_cast<_Tp>(gridDim.z)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp count_as(const grid_level&) noexcept
  {
    return static_cast<_Tp>(gridDim.x) * static_cast<_Tp>(gridDim.y) * static_cast<_Tp>(gridDim.z);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> index_as(const grid_level&) noexcept
  {
    return {static_cast<_Tp>(blockIdx.x), static_cast<_Tp>(blockIdx.y), static_cast<_Tp>(blockIdx.z)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp rank_as(const grid_level& __level) noexcept
  {
    const auto __dims = dims_as<_Tp>(__level);
    const auto __idx  = index_as<_Tp>(__level);
    return static_cast<_Tp>((__idx.z * __dims.y + __idx.y) * __dims.x + __idx.x);
  }

  // interactions with grid level in hierarchy

  _CCCL_TEMPLATE(class _Tp, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp rank_as(const grid_level& __level, const _Hierarchy& __hier) noexcept
  {
    static_assert(has_unit_or_level_v<block_level, _Hierarchy>, "_Hierarchy doesn't contain block level");
    static_assert(has_level_v<grid_level, _Hierarchy>, "_Hierarchy doesn't contain grid level");

    const auto __dims = dims_as<_Tp>(__level, __hier);
    const auto __idx  = index_as<_Tp>(__level, __hier);
    return static_cast<_Tp>((__idx.z * __dims.y + __idx.y) * __dims.x + __idx.x);
  }
#  endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_GLOBAL_CONSTANT block_level block;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_BLOCK_LEVEL_H
