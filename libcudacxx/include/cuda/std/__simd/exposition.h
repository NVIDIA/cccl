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

#include <cuda/__fwd/complex.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], exposition-only helpers

template <size_t _Bytes>
using __integer_from = __make_nbit_int_t<_Bytes * 8, true>;

// complex<T> where T is a vectorizable floating-point type
template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v = false;

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::cuda::std::complex<_Tp>> = true;

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::cuda::complex<_Tp>> = true;

#if _CCCL_HAS_HOST_STD_LIB()

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::std::complex<_Tp>> = true;

#endif // _CCCL_HAS_HOST_STD_LIB()

// [simd.expos], vectorizable types:
// all standard integer types, character types, and the types float and double ([basic.fundamental]);
// std::float16_t, std::float32_t, and std::float64_t if defined ([basic.extended.fp]); and
// complex<T> where T is a vectorizable floating-point type.
template <typename _Tp>
inline constexpr bool __is_vectorizable_v =
  (__is_extended_arithmetic_v<_Tp> || __is_complex_vectorizable_v<_Tp>)
  && !is_same_v<_Tp, bool> && !is_const_v<_Tp> && !is_volatile_v<_Tp>;

template <typename _Abi>
inline constexpr bool __is_enabled_abi_v = false;

template <__simd_size_type _Np>
inline constexpr bool __is_enabled_abi_v<__fixed_size<_Np>> = (_Np >= 1 && _Np <= 64);

template <size_t _Bytes>
inline constexpr bool __is_vectorizable_byte_size_v =
  (_Bytes == 1 || _Bytes == 2 || _Bytes == 4 || _Bytes == 8
#if _CCCL_HAS_INT128()
   || _Bytes == 16
#endif // _CCCL_HAS_INT128()
  );

// Disabled vectors remain complete even when sizeof cannot be applied to their element type
template <typename _Tp, bool = is_object_v<remove_reference_t<_Tp>>, typename = void>
struct __simd_element_size : integral_constant<size_t, 0>
{};

template <typename _Tp>
struct __simd_element_size<_Tp, true, void_t<decltype(sizeof(_Tp))>> : integral_constant<size_t, sizeof(_Tp)>
{};

template <size_t _Bytes, typename _Abi>
inline constexpr bool __is_enabled_mask_v = __is_vectorizable_byte_size_v<_Bytes> && __is_enabled_abi_v<_Abi>;

template <typename _Tp, typename _Abi>
inline constexpr bool __is_enabled_vec_v =
  __is_vectorizable_v<_Tp> && __is_enabled_mask_v<__simd_element_size<_Tp>::value, _Abi>;

// [simd.expos], simd-complex-value-type
template <typename _Tp>
using __simd_complex_value_type_t = typename _Tp::value_type;

template <typename _Tp, typename _Abi>
inline constexpr __simd_size_type __simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __simd_size_v<_Tp, fixed_size<_Np>> =
  __is_enabled_vec_v<_Tp, fixed_size<_Np>> ? _Np : 0;

template <size_t _Bytes, typename _Abi>
inline constexpr __simd_size_type __mask_size_v = 0;

template <size_t _Bytes, __simd_size_type _Np>
inline constexpr __simd_size_type __mask_size_v<_Bytes, fixed_size<_Np>> =
  __is_enabled_mask_v<_Bytes, fixed_size<_Np>> ? _Np : 0;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_EXPOSITION_H
