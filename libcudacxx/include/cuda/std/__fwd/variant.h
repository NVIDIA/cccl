//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_VARIANT_H
#define _CUDA_STD___FWD_VARIANT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT variant;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_size;

template <class _Tp>
inline constexpr size_t variant_size_v = variant_size<_Tp>::value;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT variant_alternative;

template <size_t _Ip, class _Tp>
using variant_alternative_t = typename variant_alternative<_Ip, _Tp>::type;

inline constexpr size_t variant_npos = static_cast<size_t>(-1);

template <class _IndexType>
inline constexpr _IndexType __variant_npos = static_cast<_IndexType>(-1);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_VARIANT_H
