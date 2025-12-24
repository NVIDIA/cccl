//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_CONCEPTS_H
#define _CUDAX___SIMD_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__simd/declaration.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <class _Tp>
_CCCL_CONCEPT __has_pre_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((++__t));

template <class _Tp>
_CCCL_CONCEPT __has_post_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t++));

template <class _Tp>
_CCCL_CONCEPT __has_pre_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((--__t));

template <class _Tp>
_CCCL_CONCEPT __has_post_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t--));

template <class _Tp>
_CCCL_CONCEPT __has_negate = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((!__t));

template <class _Tp>
_CCCL_CONCEPT __has_bitwise_not = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((~__t));

template <class _Tp>
_CCCL_CONCEPT __has_plus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((+__t));

template <class _Tp>
_CCCL_CONCEPT __has_unary_minus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((-__t));

template <class _Tp>
_CCCL_CONCEPT __has_minus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t - __t));

template <class _Tp>
_CCCL_CONCEPT __has_multiplies = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t * __t));

template <class _Tp>
_CCCL_CONCEPT __has_divides = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t / __t));

template <class _Tp>
_CCCL_CONCEPT __has_modulo = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t % __t));

template <class _Tp>
_CCCL_CONCEPT __has_bitwise_and = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t & __t));

template <class _Tp>
_CCCL_CONCEPT __has_bitwise_or = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t | __t));

template <class _Tp>
_CCCL_CONCEPT __has_bitwise_xor = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t ^ __t));

template <class _Tp>
_CCCL_CONCEPT __has_shift_left = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t << __t));

template <class _Tp>
_CCCL_CONCEPT __has_shift_right = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >> __t));

template <class _Tp>
_CCCL_CONCEPT __has_shift_left_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t << __simd_size_type{}));

template <class _Tp>
_CCCL_CONCEPT __has_shift_right_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >> __simd_size_type{}));

template <class _Tp>
_CCCL_CONCEPT __has_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t == __t));

template <class _Tp>
_CCCL_CONCEPT __has_not_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t != __t));

template <class _Tp>
_CCCL_CONCEPT __has_greater_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >= __t));

template <class _Tp>
_CCCL_CONCEPT __has_less_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t <= __t));

template <class _Tp>
_CCCL_CONCEPT __has_greater = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t > __t));

template <class _Tp>
_CCCL_CONCEPT __has_less = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t < __t));
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_CONCEPTS_H
