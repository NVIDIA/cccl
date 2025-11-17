// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_GROUP_TRAITS_H_
#define _CUDA___MEMCPY_ASYNC_GROUP_TRAITS_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// forward declare cooperative groups types. we cannot include <cooperative_groups.h> since it does not work with NVHPC
namespace cooperative_groups
{
namespace __v1
{
class thread_block;

template <unsigned int Size, typename ParentT>
class thread_block_tile;
} // namespace __v1
using namespace __v1;
} // namespace cooperative_groups

_CCCL_BEGIN_NAMESPACE_CUDA

//! Trait to detect whether a group represents a CUDA thread block, for example: ``cooperative_groups::thread_block``.
template <typename _Group>
inline constexpr bool is_thread_block_group_v = false;

template <>
inline constexpr bool is_thread_block_group_v<::cooperative_groups::thread_block> = true;

//! Trait to detect whether a group represents a CUDA warp, for example:
//! ``cooperative_groups::thread_block_tile<32, ...>``.
template <typename _Group>
inline constexpr bool is_warp_group_v = false;

template <typename _Parent>
inline constexpr bool is_warp_group_v<::cooperative_groups::thread_block_tile<32, _Parent>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_GROUP_TRAITS_H_
