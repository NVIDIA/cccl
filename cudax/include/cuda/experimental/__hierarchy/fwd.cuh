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

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// hierarchy group kinds

class __this_hierarchy_group_kind
{};

template <class _InLevel, ::cuda::std::size_t _Np>
class __group_by_hierarchy_group_kind
{};

template <class _InLevel>
class __group_as_hierarchy_group_kind
{};

template <class _InLevel, bool _Opt>
class __generic_hierarchy_group_kind
{};

// hierarchy group base

template <class _Level, class _Hierarchy, class _Kind>
class __hierarchy_group_base;
template <class _Level, class _Hierarchy>
using __this_hierarchy_group_base = __hierarchy_group_base<_Level, _Hierarchy, __this_hierarchy_group_kind>;

// hierarchy groups

template <class _Kind, class _Hierarchy>
class thread_group;
template <class _Kind, class _Hierarchy>
class warp_group;
template <class _Kind, class _Hierarchy>
class block_group;
template <class _Kind, class _Hierarchy>
class cluster_group;
template <class _Kind, class _Hierarchy>
class grid_group;

// traits

template <class _Tp>
inline constexpr bool __is_this_hierarchy_group_v = false;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<thread_group<__this_hierarchy_group_kind, _Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<warp_group<__this_hierarchy_group_kind, _Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<block_group<__this_hierarchy_group_kind, _Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<cluster_group<__this_hierarchy_group_kind, _Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_hierarchy_group_v<grid_group<__this_hierarchy_group_kind, _Hierarchy>> = true;

template <::cuda::std::size_t _Np>
struct group_by_t;

class group_as;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_FWD_CUH
