//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H
#define _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__hierarchy/block_level.h>
#  include <cuda/__hierarchy/cluster_level.h>
#  include <cuda/__hierarchy/grid_level.h>
#  include <cuda/__hierarchy/thread_level.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/**
 * @brief Returns a tuple of dim3 compatible objects that can be used to launch
 * a kernel
 *
 * This function returns a tuple of hierarchy_query_result objects that contain
 * dimensions from the supplied hierarchy, that can be used to launch that
 * hierarchy. It is meant to allow for easy usage of hierarchy dimensions with
 * the <<<>>> launch syntax or cudaLaunchKernelEx in case of a cluster launch.
 * Contained hierarchy_query_result objects are results of extents() member
 * function on the hierarchy passed in. The returned tuple has three elements if
 * cluster_level is present in the hierarchy (extents(block, grid),
 * extents(cluster, block), extents(thread, block)). Otherwise it contains only
 * two elements, without the middle one related to the cluster.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
 * block_dims<8, 8, 8>()); auto [grid_dimensions, cluster_dimensions,
 * block_dimensions] = get_launch_dimensions(hierarchy);
 * assert(grid_dimensions.x == 256);
 * assert(cluster_dimensions.x == 4);
 * assert(block_dimensions.x == 8);
 * assert(block_dimensions.y == 8);
 * assert(block_dimensions.z == 8);
 * @endcode
 * @par
 *
 * @param hierarchy
 *  Hierarchy that the launch dimensions are requested for
 */
template <class... _Levels>
constexpr auto _CCCL_HOST get_launch_dimensions(const hierarchy_dimensions<_Levels...>& __hierarchy)
{
  if constexpr (has_level_v<cluster_level, hierarchy_dimensions<_Levels...>>)
  {
    return ::cuda::std::make_tuple(
      __hierarchy.extents(block_level{}, grid_level{}),
      __hierarchy.extents(block_level{}, cluster_level{}),
      __hierarchy.extents(thread_level{}, block_level{}));
  }
  else
  {
    return ::cuda::std::make_tuple(
      __hierarchy.extents(block_level{}, grid_level{}), __hierarchy.extents(gpu_thread, block_level{}));
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_GET_LAUNCH_DIMENSIONS_H
