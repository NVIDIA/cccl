// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD_EXPERIMENTAL___SIMD_TRAITS_H
#define _CUDA_STD_EXPERIMENTAL___SIMD_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/experimental/__simd/config.h>
#include <cuda/std/experimental/__simd/declaration.h>
#include <cuda/std/experimental/__simd/utility.h>

#if _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#  include <cuda/std/__bit/bit_ceil.h>
#  include <cuda/std/__cstddef/size_t.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace experimental
{
inline namespace parallelism_v2
{

template <class _Tp>
inline constexpr bool is_abi_tag_v = false;

template <class _Tp>
struct is_abi_tag : bool_constant<is_abi_tag_v<_Tp>>
{};

template <class _Tp>
inline constexpr bool is_simd_v = false;

template <class _Tp>
struct is_simd : bool_constant<is_simd_v<_Tp>>
{};

template <class _Tp>
inline constexpr bool is_simd_flag_type_v = false;

template <class _Tp>
struct is_simd_flag_type : bool_constant<is_simd_flag_type_v<_Tp>>
{};

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>, bool = (__is_vectorizable_v<_Tp> && is_abi_tag_v<_Abi>)>
struct simd_size : integral_constant<size_t, _Abi::__simd_size>
{};

template <class _Tp, class _Abi>
struct simd_size<_Tp, _Abi, false>
{};

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
inline constexpr size_t simd_size_v = simd_size<_Tp, _Abi>::value;

} // namespace parallelism_v2
} // namespace experimental

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#endif // _CUDA_STD_EXPERIMENTAL___SIMD_TRAITS_H

