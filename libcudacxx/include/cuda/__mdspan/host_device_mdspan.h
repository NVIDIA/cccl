//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H
#define _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/host_device_accessor.h>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
using host_mdspan = ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, host_accessor<_AccessorPolicy>>;

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
using device_mdspan = ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, device_accessor<_AccessorPolicy>>;

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
using managed_mdspan = ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, managed_accessor<_AccessorPolicy>>;

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_host_accessible_v<::cuda::std::mdspan<_Tp, _Ep, _Lp, _Ap>> = is_host_accessible_v<_Ap>;

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_device_accessible_v<::cuda::std::mdspan<_Tp, _Ep, _Lp, _Ap>> = is_device_accessible_v<_Ap>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H
