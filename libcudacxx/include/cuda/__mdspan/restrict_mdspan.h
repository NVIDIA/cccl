//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_RESTRICT_MDSPAN
#define _CUDA___MDSPAN_RESTRICT_MDSPAN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/restrict_accessor.h>
#include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = _CUDA_VSTD::layout_right,
          typename _AccessorPolicy = _CUDA_VSTD::default_accessor<_ElementType>>
using restrict_mdspan = _CUDA_VSTD::mdspan<_ElementType, _Extents, _LayoutPolicy, restrict_accessor<_AccessorPolicy>>;

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_restrict_mdspan_v = false;

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_restrict_mdspan_v<_CUDA_VSTD::mdspan<_Tp, _Ep, _Lp, _Ap>> = is_restrict_accessor_v<_Ap>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MDSPAN_RESTRICT_MDSPAN
