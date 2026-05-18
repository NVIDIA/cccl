//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_EXPOSITION_H
#define _CUDA_STD___SIMD_EXPOSITION_H

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
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], exposition-only helpers

template <size_t _Bytes>
using __integer_from = __make_nbit_int_t<_Bytes * 8, true>;

// all standard integer types, character types, and the types float and double ([basic.fundamental]);
// std​::​float16_t, std​::​float32_t, and std​::​float64_t if defined ([basic.extended.fp]); and
// TODO(fbusato) complex<T> where T is a vectorizable floating-point type.
template <typename _Tp>
inline constexpr bool __is_vectorizable_v =
  __is_extended_arithmetic_v<_Tp> && !is_same_v<_Tp, bool> && !is_const_v<_Tp> && !is_volatile_v<_Tp>;

template <typename _Tp, typename _Abi>
inline constexpr __simd_size_type __simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __simd_size_v<_Tp, fixed_size<_Np>> = _Np;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_EXPOSITION_H
