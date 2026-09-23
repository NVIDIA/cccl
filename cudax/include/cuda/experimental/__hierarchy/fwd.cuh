//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_FWD_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_FWD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__fwd/span.h>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
using __implicit_hierarchy_t =
  hierarchy<thread_level,
            hierarchy_level_desc<grid_level, ::cuda::std::dims<3, unsigned>>,
            hierarchy_level_desc<cluster_level, ::cuda::std::dims<3, unsigned>>,
            hierarchy_level_desc<block_level, ::cuda::std::dims<3, unsigned>>>;

using __implicit_hierarchy_1d_t =
  hierarchy<thread_level,
            hierarchy_level_desc<grid_level, ::cuda::std::extents<unsigned, ::cuda::std::dynamic_extent, 1, 1>>,
            hierarchy_level_desc<cluster_level, ::cuda::std::extents<unsigned, ::cuda::std::dynamic_extent, 1, 1>>,
            hierarchy_level_desc<block_level, ::cuda::std::extents<unsigned, ::cuda::std::dynamic_extent, 1, 1>>>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_FWD_CUH
