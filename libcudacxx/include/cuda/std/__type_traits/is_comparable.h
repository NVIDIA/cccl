//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class _Up>
_CCCL_CONCEPT __is_equality_comparable_v = _CCCL_REQUIRES_EXPR((_Tp, _Up), _Tp __lhs, _Up __rhs)((__lhs == __rhs));

template <class _Tp, class _Up>
_CCCL_CONCEPT __is_nothrow_equality_comparable_v = _CCCL_REQUIRES_EXPR((_Tp, _Up), _Tp __lhs, _Up __rhs)(
  requires(__is_equality_comparable_v<_Tp, _Up>), noexcept(__lhs == __rhs));

template <class _Tp, class _Up>
_CCCL_CONCEPT __is_less_than_comparable_v = _CCCL_REQUIRES_EXPR((_Tp, _Up), _Tp __lhs, _Up __rhs)((__lhs < __rhs));

template <class _Tp, class _Up>
_CCCL_CONCEPT __is_nothrow_less_than_comparable_v = _CCCL_REQUIRES_EXPR((_Tp, _Up), _Tp __lhs, _Up __rhs)(
  requires(__is_less_than_comparable_v<_Tp, _Up>), noexcept(__lhs < __rhs));

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H
