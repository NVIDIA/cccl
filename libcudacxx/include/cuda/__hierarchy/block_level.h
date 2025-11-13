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

#include <cuda/__fwd/hierarchy.h>
#include <cuda/__hierarchy/hierarchy_query_result.h>
#include <cuda/__hierarchy/native_hierarchy_level_base.h>
#include <cuda/std/__mdspan/extents.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct block_level : __native_hierarchy_level_base<block_level>
{
  using __next_level = cluster_level;

  using __base_type = __native_hierarchy_level_base<block_level>;
  using __base_type::extents;
  using __base_type::index;

  // interactions with cluster level

  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, unsigned> extents(const cluster_level&) noexcept
  {
    ::dim3 __dims{1u, 1u, 1u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__dims = ::__clusterDim();))
    return ::cuda::std::dims<3, unsigned>{__dims.z, __dims.y, __dims.x};
  }
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<unsigned, 3> index(const cluster_level&) noexcept
  {
    ::dim3 __idx{0u, 0u, 0u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__idx = ::__clusterRelativeBlockIdx();))
    return {__idx.z, __idx.y, __idx.x};
  }

  // interactions with grid level

  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, unsigned> extents(const grid_level&) noexcept
  {
    return ::cuda::std::dims<3, unsigned>{gridDim.z, gridDim.y, gridDim.x};
  }
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<unsigned, 3> index(const grid_level&) noexcept
  {
    return {blockIdx.z, blockIdx.y, blockIdx.x};
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

_CCCL_GLOBAL_CONSTANT ::cuda::block_level block;

_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_BLOCK_LEVEL_H
