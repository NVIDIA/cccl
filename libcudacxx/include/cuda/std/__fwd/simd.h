//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_SIMD_H
#define _CUDA_STD___FWD_SIMD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/exposition.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Tp, typename _Abi = native<_Tp>, typename = void>
class basic_vec;

template <size_t _Bytes, typename _Abi = native<__integer_from<_Bytes>>, typename = void>
class basic_mask;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, native<_Tp>>>
using vec = basic_vec<_Tp, __deduce_abi_t<_Tp, _Np>>;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, native<_Tp>>>
using mask = basic_mask<sizeof(_Tp), __deduce_abi_t<_Tp, _Np>>;

// specializations

template <typename _Tp, typename _Abi>
struct __simd_storage;

template <typename _Tp, typename _Abi>
struct __simd_operations;

template <size_t _Bytes, typename _Abi>
struct __mask_storage;

template <size_t _Bytes, typename _Abi>
struct __mask_operations;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_SIMD_H
