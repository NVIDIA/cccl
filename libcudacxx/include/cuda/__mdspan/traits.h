//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_TRAITS_H
#define _CUDA___MDSPAN_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _LayoutPolicy>
inline constexpr bool __is_layout_right_v = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_right>;

template <typename _LayoutPolicy>
inline constexpr bool __is_layout_left_v = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_left>;

template <typename _LayoutPolicy>
inline constexpr bool __is_layout_stride_v = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_stride>;

template <typename _LayoutPolicy>
inline constexpr bool __is_cuda_mdspan_layout_v =
  __is_layout_right_v<_LayoutPolicy> || __is_layout_left_v<_LayoutPolicy> || __is_layout_stride_v<_LayoutPolicy>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_TRAITS_H
