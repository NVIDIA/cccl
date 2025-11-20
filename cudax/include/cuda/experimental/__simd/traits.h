//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_TRAITS_H
#define _CUDAX___SIMD_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _Tp>
inline constexpr bool is_abi_tag_v = false;

template <typename _Tp>
struct is_abi_tag : ::cuda::std::bool_constant<is_abi_tag_v<_Tp>>
{};

template <typename _Tp>
inline constexpr bool is_simd_v = false;

template <typename _Tp>
struct is_simd : ::cuda::std::bool_constant<is_simd_v<_Tp>>
{};

template <typename _Tp>
inline constexpr bool is_simd_flag_type_v = false;

template <typename _Tp>
struct is_simd_flag_type : ::cuda::std::bool_constant<is_simd_flag_type_v<_Tp>>
{};

template <typename _Tp,
          typename _Abi = simd_abi::compatible<_Tp>,
          bool          = (__is_vectorizable_v<_Tp> && is_abi_tag_v<_Abi>)>
struct simd_size : ::cuda::std::integral_constant<::cuda::std::size_t, _Abi::__simd_size>
{};

template <typename _Tp, typename _Abi>
struct simd_size<_Tp, _Abi, false>
{};

template <typename _Tp, typename _Abi = simd_abi::compatible<_Tp>>
inline constexpr ::cuda::std::size_t simd_size_v = simd_size<_Tp, _Abi>::value;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_TRAITS_H
