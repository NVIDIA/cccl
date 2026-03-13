//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_IMPLICIT_HIERARCHY_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_IMPLICIT_HIERARCHY_CUH

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
[[nodiscard]] _CCCL_DEVICE_API inline __implicit_hierarchy_t __implicit_hierarchy() noexcept
{
  return __implicit_hierarchy_t{
    cuda::gpu_thread,
    hierarchy_level_desc<grid_level, ::cuda::std::dims<3, unsigned>>{cluster.extents(grid)},
    hierarchy_level_desc<cluster_level, ::cuda::std::dims<3, unsigned>>{block.extents(cluster)},
    hierarchy_level_desc<block_level, ::cuda::std::dims<3, unsigned>>{gpu_thread.extents(block)}};
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_IMPLICIT_HIERARCHY_CUH
