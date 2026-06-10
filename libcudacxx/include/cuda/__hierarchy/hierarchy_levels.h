//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_HIERARCHY_LEVELS_H
#define _CUDA___HIERARCHY_HIERARCHY_LEVELS_H

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
#  include <cuda/__hierarchy/native_hierarchy_level_base.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct _CCCL_DECLSPEC_EMPTY_BASES thread_level : __native_hierarchy_level_base<thread_level>
{
  using __product_type  = ::cuda::std::uint32_t;
  using __allowed_above = __allowed_levels<block_level>;
  using __allowed_below = __allowed_levels<>;

  using __next_native_level = block_level;
};

struct _CCCL_DECLSPEC_EMPTY_BASES warp_level : __native_hierarchy_level_base<warp_level>
{
  using __product_type = ::cuda::std::uint32_t;

  using __next_native_level = block_level;
};

struct _CCCL_DECLSPEC_EMPTY_BASES block_level : __native_hierarchy_level_base<block_level>
{
  using __product_type  = ::cuda::std::uint32_t;
  using __allowed_above = __allowed_levels<grid_level, cluster_level>;
  using __allowed_below = __allowed_levels<thread_level>;

  using __next_native_level = cluster_level;
};

struct _CCCL_DECLSPEC_EMPTY_BASES cluster_level : __native_hierarchy_level_base<cluster_level>
{
  using __product_type  = ::cuda::std::uint32_t;
  using __allowed_above = __allowed_levels<grid_level>;
  using __allowed_below = __allowed_levels<block_level>;

  using __next_native_level = grid_level;
};

struct _CCCL_DECLSPEC_EMPTY_BASES grid_level : __native_hierarchy_level_base<grid_level>
{
  using __product_type  = ::cuda::std::uint64_t;
  using __allowed_above = __allowed_levels<>;
  using __allowed_below = __allowed_levels<block_level, cluster_level>;
};

_CCCL_GLOBAL_CONSTANT thread_level gpu_thread;
_CCCL_GLOBAL_CONSTANT warp_level warp;
_CCCL_GLOBAL_CONSTANT block_level block;
_CCCL_GLOBAL_CONSTANT cluster_level cluster;
_CCCL_GLOBAL_CONSTANT grid_level grid;

// Struct to represent levels allowed below or above a certain level,
//  used for hierarchy sorting, validation and for hierarchy traversal
template <typename... _Levels>
struct __allowed_levels
{
  using __default_unit = ::cuda::std::__type_index_c<0, _Levels..., void>;
};

namespace __detail
{
template <typename LevelType>
using __default_unit_below = typename LevelType::__allowed_below::__default_unit;

template <class _QueryLevel, class _AllowedLevels>
inline constexpr bool __is_level_allowed = false;

template <class _QueryLevel, class... _Levels>
inline constexpr bool __is_level_allowed<_QueryLevel, __allowed_levels<_Levels...>> =
  (::cuda::std::is_same_v<_QueryLevel, _Levels> || ...);

template <class _L1, class _L2>
inline constexpr bool __can_rhs_stack_on_lhs =
  __is_level_allowed<_L1, typename _L2::__allowed_below> || __is_level_allowed<_L2, typename _L1::__allowed_above>;

template <class _Unit, class _Level>
inline constexpr bool __legal_unit_for_level =
  __can_rhs_stack_on_lhs<_Unit, _Level> || __legal_unit_for_level<_Unit, __default_unit_below<_Level>>;

template <class _Unit>
inline constexpr bool __legal_unit_for_level<_Unit, void> = false;
} // namespace __detail

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_LEVELS_H
