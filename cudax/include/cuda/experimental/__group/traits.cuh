//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Tp, class = void>
inline constexpr bool __is_spannable = false;
template <class _Tp>
inline constexpr bool
  __is_spannable<_Tp, ::cuda::std::void_t<decltype(::cuda::std::span(::cuda::std::declval<_Tp>()))>> = true;

template <class _Span>
using _SpanElementType = typename _Span::element_type;

template <class _Span>
using _SpanValueType = typename _Span::value_type;

template <class _Unit, class _Level>
inline constexpr bool __unit_same_as_or_below_v = __is_natively_reachable_hierarchy_level_v<_Unit, _Level>;
template <class _Level>
inline constexpr bool __unit_same_as_or_below_v<_Level, _Level> = true;
template <>
inline constexpr bool __unit_same_as_or_below_v<thread_level, warp_level> = true;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
