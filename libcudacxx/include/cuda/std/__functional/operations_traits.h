//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H
#define _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HOSTED()
_CCCL_BEGIN_NAMESPACE_STD

template <typename>
struct plus;
template <typename>
struct multiplies;
template <typename>
struct bit_and;
template <typename>
struct bit_or;
template <typename>
struct bit_xor;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HOSTED()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _BinaryOp>
inline constexpr bool __is_plus_op_v =
  is_same_v<_BinaryOp, plus<>>
#if _CCCL_HOSTED()
  || is_same_v<_BinaryOp, ::std::plus<void>>
#endif // _CCCL_HOSTED()
  ;

template <typename _BinaryOp>
inline constexpr bool __is_multiplies_op_v =
  is_same_v<_BinaryOp, multiplies<>>
#if _CCCL_HOSTED()
  || is_same_v<_BinaryOp, ::std::multiplies<void>>
#endif // _CCCL_HOSTED()
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_and_op_v =
  is_same_v<_BinaryOp, bit_and<>>
#if _CCCL_HOSTED()
  || is_same_v<_BinaryOp, ::std::bit_and<void>>
#endif // _CCCL_HOSTED()
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_or_op_v =
  is_same_v<_BinaryOp, bit_or<>>
#if _CCCL_HOSTED()
  || is_same_v<_BinaryOp, ::std::bit_or<void>>
#endif // _CCCL_HOSTED()
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_xor_op_v =
  is_same_v<_BinaryOp, bit_xor<>>
#if _CCCL_HOSTED()
  || is_same_v<_BinaryOp, ::std::bit_xor<void>>
#endif // _CCCL_HOSTED()
  ;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H
