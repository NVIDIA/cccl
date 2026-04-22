//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_TYPE_TRAITS_H
#define _CUDA_STD___SIMD_TYPE_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_valid_alignment.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.traits], alignment
template <typename _Tp, typename _Up = typename _Tp::value_type>
struct alignment;

template <typename _Tp, typename _Abi, typename _Up>
struct alignment<basic_vec<_Tp, _Abi>, _Up>
    : integral_constant<size_t,
                        ::cuda::__is_valid_alignment(__simd_size_v<_Tp, _Abi> * alignof(_Up))
                          ? __simd_size_v<_Tp, _Abi> * alignof(_Up)
                          : alignof(_Up)>
{
  static_assert(__is_vectorizable_v<_Up>, "U must be a vectorizable type");
};

template <typename _Tp, typename _Up = typename _Tp::value_type>
inline constexpr size_t alignment_v = alignment<_Tp, _Up>::value;

// [simd.traits], rebind
template <typename _Tp, typename _Vp>
struct rebind;

template <typename _Tp, typename _Up, typename _Abi>
struct rebind<_Tp, basic_vec<_Up, _Abi>>
{
  static_assert(__is_vectorizable_v<_Tp>, "T must be a vectorizable type");
  using type = basic_vec<_Tp, __deduce_abi_t<_Tp, __simd_size_v<_Up, _Abi>>>;
};

template <typename _Tp, size_t _Bytes, typename _Abi>
struct rebind<_Tp, basic_mask<_Bytes, _Abi>>
{
  static_assert(__is_vectorizable_v<_Tp>, "T must be a vectorizable type");
  using __integer_t       = __integer_from<sizeof(_Tp)>;
  using __integer_bytes_t = __integer_from<_Bytes>;

  using type = basic_mask<sizeof(_Tp), __deduce_abi_t<__integer_t, __simd_size_v<__integer_bytes_t, _Abi>>>;
};

template <typename _Tp, typename _Vp>
using rebind_t = typename rebind<_Tp, _Vp>::type;

// [simd.traits], resize
template <__simd_size_type _Np, typename _Vp>
struct resize;

template <__simd_size_type _Np, typename _Tp, typename _Abi>
struct resize<_Np, basic_vec<_Tp, _Abi>>
{
  using type = basic_vec<_Tp, __deduce_abi_t<_Tp, _Np>>;
};

template <__simd_size_type _Np, size_t _Bytes, typename _Abi>
struct resize<_Np, basic_mask<_Bytes, _Abi>>
{
  using type = basic_mask<_Bytes, __deduce_abi_t<__integer_from<_Bytes>, _Np>>;
};

template <__simd_size_type _Np, typename _Vp>
using resize_t = typename resize<_Np, _Vp>::type;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_TYPE_TRAITS_H
