//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_DECLARATION_H
#define _CUDAX___SIMD_DECLARATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
// exposition-only helpers
using __simd_size_type = ::cuda::std::ptrdiff_t;

template <::cuda::std::size_t _Bytes>
using __integer_from_bytes = ::cuda::std::__make_nbit_int_t<_Bytes * 8, false>;

namespace simd_abi
{
struct __scalar;

using scalar = __scalar;

template <int _Np>
struct __fixed_size;

template <int _Np>
using fixed_size = __fixed_size<_Np>;

template <typename>
using compatible = fixed_size<1>;

template <typename>
using native = fixed_size<1>;

template <typename T, __simd_size_type _Np>
using deduce = fixed_size<_Np>;
} // namespace simd_abi

// exposition-only helpers
template <typename _Tp, typename _Abi>
inline constexpr __simd_size_type __simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __simd_size_v<_Tp, simd_abi::fixed_size<_Np>> = _Np;

template <typename _Tp>
inline constexpr __simd_size_type __simd_size_v<_Tp, simd_abi::native<_Tp>> = 1;

template <typename _Tp, typename _Abi>
struct __simd_storage;

template <typename _Tp, typename _Abi>
struct __simd_operations;

template <::cuda::std::size_t _Bytes, typename _Abi>
struct __mask_storage;

template <::cuda::std::size_t _Bytes, typename _Abi>
struct __mask_operations;

// P1928R15: basic_vec is the primary SIMD vector type
template <typename _Tp, typename _Abi = simd_abi::native<_Tp>>
class basic_vec;

// P1928R15: basic_mask is the primary SIMD mask type with Bytes as first template parameter
template <::cuda::std::size_t _Bytes, typename _Abi = simd_abi::native<__integer_from_bytes<_Bytes>>>
class basic_mask;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, simd_abi::native<_Tp>>>
using vec = basic_vec<_Tp, simd_abi::deduce<_Tp, _Np>>;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, simd_abi::native<_Tp>>>
using mask = basic_mask<sizeof(_Tp), simd_abi::deduce<_Tp, _Np>>;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_DECLARATION_H
