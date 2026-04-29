//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_ABI_H
#define _CUDA_STD___SIMD_ABI_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

using __simd_size_type = ptrdiff_t;

// [simd.expos.abi], simd ABI tags
template <__simd_size_type _Np>
struct __fixed_size; // internal ABI tag

template <__simd_size_type _Np>
using fixed_size = __fixed_size<_Np>; // implementation-defined ABI

// TODO(fbusato): this could be optimized by using max access size / sizeof(T)
template <typename>
using native = fixed_size<1>; // implementation-defined ABI

template <typename, __simd_size_type _Np>
using __deduce_abi_t = fixed_size<_Np>; // exposition-only

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_ABI_H
