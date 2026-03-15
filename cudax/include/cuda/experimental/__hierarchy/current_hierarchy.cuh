//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_CURRENT_HIERARCHY_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_CURRENT_HIERARCHY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>

#include <cuda/experimental/__hierarchy/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API _Hierarchy __current_hierarchy() noexcept
{
  using _GridDesc = typename _Hierarchy::template level_desc_type<grid_level>;
  using _GridExts = typename _GridDesc::extents_type;

  using _BlockDesc = typename _Hierarchy::template level_desc_type<block_level>;
  using _BlockExts = typename _BlockDesc::extents_type;

  if constexpr (_Hierarchy::has_level(cluster))
  {
    using _ClusterDesc = typename _Hierarchy::template level_desc_type<cluster_level>;
    using _ClusterExts = typename _ClusterDesc::extents_type;

    return _Hierarchy{cuda::gpu_thread,
                      _GridDesc{_GridExts{cluster.extents(grid)}},
                      _ClusterDesc{_ClusterExts{block.extents(cluster)}},
                      _BlockDesc{_BlockExts{gpu_thread.extents(block)}}};
  }
  else
  {
    return _Hierarchy{
      cuda::gpu_thread, _GridDesc{_GridExts{cluster.extents(grid)}}, _BlockDesc{_BlockExts{gpu_thread.extents(block)}}};
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_CURRENT_HIERARCHY_CUH
