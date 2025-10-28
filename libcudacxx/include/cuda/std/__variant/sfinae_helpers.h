//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_SFINAE_HELPERS_H
#define _CUDA_STD___VARIANT_SFINAE_HELPERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/sfinae_helpers.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <bool _CanCopy>
struct __variant_copy_base
{};
template <>
struct __variant_copy_base<false>
{
  _CCCL_HIDE_FROM_ABI __variant_copy_base()                                      = default;
  __variant_copy_base(__variant_copy_base const&)                                = delete;
  _CCCL_HIDE_FROM_ABI __variant_copy_base(__variant_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __variant_copy_base& operator=(__variant_copy_base const&) = default;
  _CCCL_HIDE_FROM_ABI __variant_copy_base& operator=(__variant_copy_base&&)      = default;
};

template <bool _CanCopy, bool _CanMove>
struct __variant_move_base : __variant_copy_base<_CanCopy>
{};
template <bool _CanCopy>
struct __variant_move_base<_CanCopy, false> : __variant_copy_base<_CanCopy>
{
  _CCCL_HIDE_FROM_ABI __variant_move_base()                                      = default;
  _CCCL_HIDE_FROM_ABI __variant_move_base(__variant_move_base const&)            = default;
  __variant_move_base(__variant_move_base&&)                                     = delete;
  _CCCL_HIDE_FROM_ABI __variant_move_base& operator=(__variant_move_base const&) = default;
  _CCCL_HIDE_FROM_ABI __variant_move_base& operator=(__variant_move_base&&)      = default;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign>
struct __variant_copy_assign_base : __variant_move_base<_CanCopy, _CanMove>
{};
template <bool _CanCopy, bool _CanMove>
struct __variant_copy_assign_base<_CanCopy, _CanMove, false> : __variant_move_base<_CanCopy, _CanMove>
{
  _CCCL_HIDE_FROM_ABI __variant_copy_assign_base()                                        = default;
  _CCCL_HIDE_FROM_ABI __variant_copy_assign_base(__variant_copy_assign_base const&)       = default;
  _CCCL_HIDE_FROM_ABI __variant_copy_assign_base(__variant_copy_assign_base&&)            = default;
  __variant_copy_assign_base& operator=(__variant_copy_assign_base const&)                = delete;
  _CCCL_HIDE_FROM_ABI __variant_copy_assign_base& operator=(__variant_copy_assign_base&&) = default;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign, bool _CanMoveAssign>
struct __variant_move_assign_base : __variant_copy_assign_base<_CanCopy, _CanMove, _CanCopyAssign>
{};
template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign>
struct __variant_move_assign_base<_CanCopy, _CanMove, _CanCopyAssign, false>
    : __variant_copy_assign_base<_CanCopy, _CanMove, _CanCopyAssign>
{
  _CCCL_HIDE_FROM_ABI __variant_move_assign_base()                                             = default;
  _CCCL_HIDE_FROM_ABI __variant_move_assign_base(__variant_move_assign_base const&)            = default;
  _CCCL_HIDE_FROM_ABI __variant_move_assign_base(__variant_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __variant_move_assign_base& operator=(__variant_move_assign_base const&) = default;
  __variant_move_assign_base& operator=(__variant_move_assign_base&&)                          = delete;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign, bool _CanMoveAssign>
using __variant_base = __variant_move_assign_base<_CanCopy, _CanMove, _CanCopyAssign, _CanMoveAssign>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_SFINAE_HELPERS_H
