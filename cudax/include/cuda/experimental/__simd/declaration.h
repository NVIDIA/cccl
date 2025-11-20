//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___SIMD_DECLARATION_H
#define _CUDA_EXPERIMENTAL___SIMD_DECLARATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/size_t.h>
#include <cuda/std/experimental/__simd/config.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
namespace simd_abi
{
struct vector_abi;

template <int _Np>
struct __fixed_size;

template <int _Np>
using fixed_size = __fixed_size<_Np>;
} // namespace simd_abi

template <typename _Tp, typename _Abi>
struct __simd_storage;

template <typename _Tp, typename _Abi>
struct __simd_operations;

template <class _Tp, typename _Abi>
struct __mask_storage;

template <class _Tp, typename _Abi>
struct __mask_operations;

template <typename _Tp, int _Np>
class simd;

template <typename _Tp, typename _Abi>
class basic_simd;

template <class _Tp, int _Np>
class simd_mask;

template <class _Tp, typename _Abi>
class basic_simd_mask;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___SIMD_DECLARATION_H
