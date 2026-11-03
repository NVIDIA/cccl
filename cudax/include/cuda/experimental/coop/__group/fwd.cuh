//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_FWD_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_FWD_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if __has_include(<cooperative_groups.h>) && !defined(_CUDAX_DISABLE_CG_INTEROP)
#  define _CCCL_HAS_COOPERATIVE_GROUPS() 1
#else // ^^^ has cooperative groups ^^^ / vvv no cooperative groups vvv
#  define _CCCL_HAS_COOPERATIVE_GROUPS() 0
#endif // ^^^ no cooperative groups ^^^

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/cstdint>

#if _CCCL_HAS_COOPERATIVE_GROUPS()
// We don't need to include the whole <cooperative_groups.h> header, we can just forward declare the CG group classes in
// the CG namespace. However, the CG namespace is versioned, so we still need to include the minimal header.
#  include <cooperative_groups/details/info.h>
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
// groups

template <class _Level, class _Hierarchy>
class __this_group_base;

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

template <class _Unit, class _ParentGroup, class _MappingResult, class _SynchronizerInstance>
class generic_group;

template <class _Unit, class _ParentGroup, class _MappingResult>
class virtual_group;

template <class _Hierarchy>
class coalesced_group;

template <class _Unit, class _Group>
class group_view;

// mappings

template <class _Fn>
class binary_partition;

template <::cuda::std::size_t _StaticUnitCount>
class group_by;

template <class _RangeView>
class group_as;

class identity_mapping;

template <::cuda::std::size_t _StaticUnitCount = ::cuda::std::dynamic_extent>
class take;

// synchronizers

template <class _Barrier, ::cuda::std::size_t _Np>
class barrier_synchronizer;

template <class _RangeView>
class interwarp_synchronizer;

class lane_synchronizer;

class level_synchronizer;

// traits

template <class _Tp>
inline constexpr bool __is_this_group_v = false;
template <class _Hierarchy>
inline constexpr bool __is_this_group_v<this_thread<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_group_v<this_warp<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_group_v<this_block<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_group_v<this_cluster<_Hierarchy>> = true;
template <class _Hierarchy>
inline constexpr bool __is_this_group_v<this_grid<_Hierarchy>> = true;

template <class _Tp>
inline constexpr bool __is_group_mapping_v = false;
template <class _Fn>
inline constexpr bool __is_group_mapping_v<binary_partition<_Fn>> = true;
template <::cuda::std::size_t _StaticUnitCount>
inline constexpr bool __is_group_mapping_v<group_by<_StaticUnitCount>> = true;
template <class _RangeView>
inline constexpr bool __is_group_mapping_v<group_as<_RangeView>> = true;
template <>
inline constexpr bool __is_group_mapping_v<identity_mapping> = true;
template <::cuda::std::size_t _StaticUnitCount>
inline constexpr bool __is_group_mapping_v<take<_StaticUnitCount>> = true;

// tags

struct non_exhaustive_t;

inline constexpr ::cuda::std::uint32_t __invalid_count_or_rank = 0xffff'ffff;
} // namespace cuda::experimental::coop

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
_CG_BEGIN_NAMESPACE

class thread_group;
class grid_group;
#    if defined(_CG_HAS_CLUSTER_GROUP)
class cluster_group;
#    endif // _CG_HAS_CLUSTER_GROUP
class thread_block;
class coalesced_group;
template <unsigned _Size, typename _ParentT>
class thread_block_tile;

_CG_END_NAMESPACE
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_FWD_CUH
