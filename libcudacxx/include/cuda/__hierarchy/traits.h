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

template <class _Level>
using __level_type_of = typename _Level::level_type;

// has_unit_v

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool __has_unit_helper_v = false;
template <class _QueryLevel, class... _Levels>
inline constexpr bool __has_unit_helper_v<_QueryLevel, hierarchy_dimensions<_QueryLevel, _Levels...>> = true;

// has_level_v

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool __has_level_helper_v = false;
template <class _QueryLevel, class _Unit, class... _Levels>
inline constexpr bool __has_level_helper_v<_QueryLevel, hierarchy_dimensions<_Unit, _Levels...>> =
  (::cuda::std::is_same_v<_QueryLevel, typename _Levels::level_type> || ...);

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool has_level_v = __has_level_helper_v<_QueryLevel, ::cuda::std::remove_cvref_t<_Hierarchy>>;

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool has_unit_v = __has_unit_helper_v<_QueryLevel, ::cuda::std::remove_cvref_t<_Hierarchy>>;

// has_unit_or_level_v

template <class _QueryLevel, class _Hierarchy>
inline constexpr bool has_unit_or_level_v = has_unit_v<_QueryLevel, _Hierarchy> || has_level_v<_QueryLevel, _Hierarchy>;

// __next_hierarchy_level

template <class _Level, class _Hierarchy>
struct __next_hierarchy_level;

template <class _Level, class _BottomUnit, class... _Levels>
struct __next_hierarchy_level<_Level, hierarchy_dimensions<_BottomUnit, _Levels...>>
{
  static constexpr ::cuda::std::size_t __level_idx =
    ::cuda::std::__find_exactly_one_t<_Level, typename _Levels::level_type...>::value;
  using __type = ::cuda::std::__type_index_c<__level_idx - 1, typename _Levels::level_type...>;
};

template <class _Level, class... _Levels>
struct __next_hierarchy_level<_Level, hierarchy_dimensions<_Level, _Levels...>>
{
  using __type = ::cuda::std::__type_index_c<(sizeof...(_Levels) - 1), typename _Levels::level_type...>;
};

template <class _Level, class _Hierarchy>
using __next_hierarchy_level_t = typename __next_hierarchy_level<_Level, _Hierarchy>::__type;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_TRAITS_H
