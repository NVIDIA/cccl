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

#include <cuda/__functional/operator_properties.h>
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
inline constexpr bool __is_plus_op_v = ::cuda::__is_cuda_std_plus_v<_BinaryOp>;

#if _CCCL_HOSTED()
template <typename... _Tp>
inline constexpr bool __is_plus_op_v<::std::plus<_Tp...>> = true;
#endif // _CCCL_HOSTED()

template <typename _BinaryOp>
inline constexpr bool __is_multiplies_op_v = ::cuda::__is_cuda_std_multiplies_v<_BinaryOp>;

#if _CCCL_HOSTED()
template <typename... _Tp>
inline constexpr bool __is_multiplies_op_v<::std::multiplies<_Tp...>> = true;
#endif // _CCCL_HOSTED()

template <typename _BinaryOp>
inline constexpr bool __is_bit_and_op_v = ::cuda::__is_cuda_std_bit_and_v<_BinaryOp>;

#if _CCCL_HOSTED()
template <typename... _Tp>
inline constexpr bool __is_bit_and_op_v<::std::bit_and<_Tp...>> = true;
#endif // _CCCL_HOSTED()

template <typename _BinaryOp>
inline constexpr bool __is_bit_or_op_v = ::cuda::__is_cuda_std_bit_or_v<_BinaryOp>;

#if _CCCL_HOSTED()
template <typename... _Tp>
inline constexpr bool __is_bit_or_op_v<::std::bit_or<_Tp...>> = true;
#endif // _CCCL_HOSTED()

template <typename _BinaryOp>
inline constexpr bool __is_bit_xor_op_v = ::cuda::__is_cuda_std_bit_xor_v<_BinaryOp>;

#if _CCCL_HOSTED()
template <typename... _Tp>
inline constexpr bool __is_bit_xor_op_v<::std::bit_xor<_Tp...>> = true;
#endif // _CCCL_HOSTED()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H
