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

// Q: Do we want to enable this by default, or do we want the user to define some macro to get the interoperability with
//    cooperative groups?
#if __has_include(<cooperative_groups.h>)
#  define _CCCL_HAS_COOPERATIVE_GROUPS() 1
#else // ^^^ has cooperative groups ^^^ / vvv no cooperative groups vvv
#  define _CCCL_HAS_COOPERATIVE_GROUPS() 0
#endif // ^^^ no cooperative groups ^^^

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__fwd/extents.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
using __implicit_hierarchy_t =
  hierarchy<thread_level,
            hierarchy_level_desc<grid_level, ::cuda::std::dims<3, unsigned>>,
            hierarchy_level_desc<cluster_level, ::cuda::std::dims<3, unsigned>>,
            hierarchy_level_desc<block_level, ::cuda::std::dims<3, unsigned>>>;

// group bases

template <class _Level, class _Hierarchy>
class __this_group_base;

// groups

template <class _Hierarchy>
class this_thread;
template <class _Hierarchy>
class this_warp;
template <class _Hierarchy>
class this_block;
template <class _Hierarchy>
class this_cluster;
template <class _Hierarchy>
class this_grid;

// traits

template <class _Tp>
inline constexpr bool __is_this_hierarchy_group_v = false;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<this_thread<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<this_warp<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<this_block<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<this_cluster<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<this_grid<_Hierarchy>> = true;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_FWD_CUH
