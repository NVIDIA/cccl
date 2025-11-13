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

#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// __is_natively_reachable_hierarchy_level_v

template <class _FromLevel, class _CurrLevel, class _ToLevel, class = void>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v = false;
template <class _FromLevel, class _CurrLevel, class _ToLevel>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<
  _FromLevel,
  _CurrLevel,
  _ToLevel,
  ::cuda::std::void_t<typename _CurrLevel::__next_level>> =
  __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, typename _CurrLevel::__next_level, _ToLevel>;
template <class _Level, class _ToLevel>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<_Level, _Level, _ToLevel> = false;
template <class _FromLevel, class _Level>
inline constexpr bool __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, _Level, _Level> = true;

template <class _FromLevel, class _ToLevel, class = void>
inline constexpr bool __is_natively_reachable_hierarchy_level_v = false;
template <class _FromLevel, class _ToLevel>
inline constexpr bool
  __is_natively_reachable_hierarchy_level_v<_FromLevel, _ToLevel, ::cuda::std::void_t<typename _FromLevel::__next_level>> =
    __is_native_hierarchy_level_v<_ToLevel>
    && __is_natively_reachable_hierarchy_level_helper_v<_FromLevel, typename _FromLevel::__next_level, _ToLevel>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_TRAITS_H
