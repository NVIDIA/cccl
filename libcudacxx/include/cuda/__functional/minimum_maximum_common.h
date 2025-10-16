//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H
#define _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H

#include <cuda/std/detail/__config>

#include "cuda/std/__utility/declval.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename _Up>
constexpr bool __is_max_min_noexcept_v = noexcept(
  (::cuda::std::declval<_Tp>() < ::cuda::std::declval<_Up>())
    ? ::cuda::std::declval<_Up>()
    : ::cuda::std::declval<_Tp>());

template <typename _Up>
constexpr bool __is_max_min_noexcept_v<__half, _Up> = true;

template <typename _Up>
constexpr bool __is_max_min_noexcept_v<__nv_bfloat16, _Up> = true;

template <typename _Tp>
constexpr bool __is_max_min_noexcept_v<_Tp, __half> = true;

template <typename _Tp>
constexpr bool __is_max_min_noexcept_v<_Tp, __nv_bfloat16> = true;

template <>
constexpr bool __is_max_min_noexcept_v<__half, __half> = true;

template <>
constexpr bool __is_max_min_noexcept_v<__nv_bfloat16, __nv_bfloat16> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H
