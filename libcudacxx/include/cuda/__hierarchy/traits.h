//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_TRAITS_H
#define _CUDA___HIERARCHY_TRAITS_H

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
#  include <cuda/std/__tuple_dir/get.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// __is_natively_reachable_hierarchy_level_v

template <class _FromLevel, class _CurrLevel, class _ToLevel, class = void>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v = false;
template <class _FromLevel, class _CurrLevel, class _ToLevel>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<
  _FromLevel,
  _CurrLevel,
  _ToLevel,
  ::cuda::std::void_t<typename _CurrLevel::__next_native_level>> =
  __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, typename _CurrLevel::__next_native_level, _ToLevel>;
template <class _Level, class _ToLevel>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<_Level, _Level, _ToLevel> = false;
template <class _FromLevel, class _Level>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, _Level, _Level> = true;

template <class _FromLevel, class _ToLevel, class = void>
inline constexpr bool __is_natively_reachable_hierarchy_level_v = false;
template <class _FromLevel, class _ToLevel>
inline constexpr bool __is_natively_reachable_hierarchy_level_v<
  _FromLevel,
  _ToLevel,
  ::cuda::std::void_t<typename _FromLevel::__next_native_level>> =
  __is_native_hierarchy_level_v<_ToLevel>
  && __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, typename _FromLevel::__next_native_level, _ToLevel>;

// __level_type_of

template <class _LevelDesc>
using __level_type_of = typename _LevelDesc::level_type;

// __has_bottom_unit_or_level_v

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool __has_bottom_unit_or_level_v =
  ::cuda::std::is_same_v<_QueryLevel, typename _Hierarchy::bottom_unit_type>
  || _Hierarchy::template has_level<_QueryLevel>();

// __next_hierarchy_level

template <class _Level, class _Hierarchy>
struct __next_hierarchy_level;

template <class _Level, class _BottomUnit, class... _LevelDescs>
struct __next_hierarchy_level<_Level, hierarchy<_BottomUnit, _LevelDescs...>>
{
  static constexpr ::cuda::std::size_t __level_idx =
    hierarchy<_BottomUnit, _LevelDescs...>::template __level_idx<_Level>;
  using __type = ::cuda::std::__type_index_c<__level_idx - 1, typename _LevelDescs::level_type...>;
};

template <class _Level, class... _LevelDescs>
struct __next_hierarchy_level<_Level, hierarchy<_Level, _LevelDescs...>>
{
  using __type = ::cuda::std::__type_index_c<(sizeof...(_LevelDescs) - 1), typename _LevelDescs::level_type...>;
};

template <class _Level, class _Hierarchy>
using __next_hierarchy_level_t = typename __next_hierarchy_level<_Level, _Hierarchy>::__type;

template <class _Type>
_CCCL_CONCEPT_FRAGMENT(__has_hierarchy_member_,
                       requires(const _Type& __instance)(requires(
                         ::cuda::__is_hierarchy_v<::cuda::std::remove_cvref_t<decltype(__instance.hierarchy())>>)));
template <class _Type>
_CCCL_CONCEPT __has_hierarchy_member = _CCCL_FRAGMENT(__has_hierarchy_member_, _Type);

template <class _Type>
inline constexpr bool __is_or_has_hierarchy_member_v = __has_hierarchy_member<_Type> || __is_hierarchy_v<_Type>;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_TRAITS_H
