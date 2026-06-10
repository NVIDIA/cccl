//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_TRAITS_HPP
#define _CUDAX___CUCO_TRAITS_HPP

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/device_reference.h>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Trait indicating if a type is tuple-like.
//!
//! @tparam _Tp Type to inspect
//! @tparam _Enable SFINAE hook
template <class _Tp, class _Enable = void>
struct is_tuple_like : ::cuda::std::false_type
{};

template <class _Tp>
struct is_tuple_like<_Tp, ::cuda::std::void_t<decltype(::cuda::std::tuple_size<_Tp>::value)>> : ::cuda::std::true_type
{};

template <class _Tp>
inline constexpr bool is_tuple_like_v = is_tuple_like<_Tp>::value;

namespace __detail
{
//! @brief Detects cuda::std pair-like types.
//!
//! @tparam _Tp Type to inspect
//! @tparam _Enable SFINAE hook
template <class _Tp, class _Enable = void>
struct __is_pair_like_impl : ::cuda::std::false_type
{};

template <class _Tp>
struct __is_pair_like_impl<_Tp,
                           ::cuda::std::void_t<decltype(::cuda::std::get<0>(::cuda::std::declval<_Tp>())),
                                               decltype(::cuda::std::get<1>(::cuda::std::declval<_Tp>())),
                                               decltype(::cuda::std::tuple_size<_Tp>::value)>>
    : ::cuda::std::conditional_t<::cuda::std::tuple_size<_Tp>::value == 2, ::cuda::std::true_type, ::cuda::std::false_type>
{};

template <class _Tp>
struct __is_pair_like
    : __is_pair_like_impl<
        ::cuda::std::remove_reference_t<decltype(thrust::raw_reference_cast(::cuda::std::declval<_Tp>()))>>
{};

template <class _Tp>
using __is_pair_like_t = __is_pair_like<_Tp>;
} // namespace __detail
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_TRAITS_HPP
