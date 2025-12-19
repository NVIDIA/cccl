//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FWD_HIERARCHY_H
#define _CUDA___FWD_HIERARCHY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_base_of.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// hierarchy level

template <class _Level>
struct hierarchy_level_base;

template <class _Level>
struct __native_hierarchy_level_base;

struct grid_level;
struct cluster_level;
struct block_level;
struct warp_level;
struct thread_level;

template <class _Tp>
inline constexpr bool __is_hierarchy_level_v = ::cuda::std::is_base_of_v<hierarchy_level_base<_Tp>, _Tp>;

template <class _Tp>
inline constexpr bool __is_native_hierarchy_level_v =
  ::cuda::std::is_base_of_v<__native_hierarchy_level_base<_Tp>, _Tp>;

// hierarchy

template <class _BottomUnit, class... _Levels>
struct hierarchy_dimensions;

template <class _Tp>
inline constexpr bool __is_hierarchy_v = false;
template <class _BottomUnit, class... _Levels>
inline constexpr bool __is_hierarchy_v<hierarchy_dimensions<_BottomUnit, _Levels...>> = true;

template <typename... _Levels>
struct allowed_levels;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FWD_HIERARCHY_H
