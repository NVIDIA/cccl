//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_IN_PLACE_H
#define _CUDA_STD___UTILITY_IN_PLACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_t() = default;
};
_CCCL_GLOBAL_CONSTANT in_place_t in_place{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_type_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_type_t() = default;
};
template <class _Tp>
inline constexpr in_place_type_t<_Tp> in_place_type{};

template <size_t _Idx>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_index_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_index_t() = default;
};
template <size_t _Idx>
inline constexpr in_place_index_t<_Idx> in_place_index{};

template <class _Tp>
inline constexpr bool __is_cuda_std_inplace_type_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_std_inplace_type_v<in_place_type_t<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_cuda_std_inplace_index_v = false;
template <size_t _Idx>
inline constexpr bool __is_cuda_std_inplace_index_v<in_place_index_t<_Idx>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

// CCCL extensions below
_CCCL_BEGIN_NAMESPACE_CUDA

struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_from_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_from_t() = default;
};
_CCCL_GLOBAL_CONSTANT in_place_from_t in_place_from{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_from_type_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_from_type_t() = default;
};
template <class _Tp>
inline constexpr in_place_from_type_t<_Tp> in_place_from_type{};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_IN_PLACE_H
