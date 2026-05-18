//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TYPE_TRAITS_VECTOR_SIZE_H
#define _CUDA___TYPE_TRAITS_VECTOR_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class _Tp>
inline constexpr ::cuda::std::size_t vector_size_v = 0;
template <class _Tp>
inline constexpr ::cuda::std::size_t vector_size_v<const _Tp> = vector_size_v<_Tp>;
template <class _Tp>
inline constexpr ::cuda::std::size_t vector_size_v<volatile _Tp> = vector_size_v<_Tp>;
template <class _Tp>
inline constexpr ::cuda::std::size_t vector_size_v<const volatile _Tp> = vector_size_v<_Tp>;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::char1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::char2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::char3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::char4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uchar1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uchar2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uchar3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uchar4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::short1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::short2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::short3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::short4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ushort1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ushort2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ushort3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ushort4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::int1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::int2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::int3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::int4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uint1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uint2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uint3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::uint4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long4> = 4;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long4_16a> = 4;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::long4_32a> = 4;
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong4> = 4;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong4_16a> = 4;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulong4_32a> = 4;
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong4> = 4;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong4_16a> = 4;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::longlong4_32a> = 4;
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong4> = 4;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong4_16a> = 4;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::ulonglong4_32a> = 4;
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::float1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::float2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::float3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::float4> = 4;

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double1> = 1;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double3> = 3;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double4> = 4;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double4_16a> = 4;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::double4_32a> = 4;
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline constexpr ::cuda::std::size_t vector_size_v<::dim3> = 3;

#  if _CCCL_HAS_NVFP16()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__half2> = 2;
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_bfloat162> = 2;
#  endif // _CCCL_HAS_NVBF16()

#  if _CCCL_HAS_NVFP8_E4M3()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x2_e4m3> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x4_e4m3> = 4;
#  endif // _CCCL_HAS_NVFP8_E4M3()

#  if _CCCL_HAS_NVFP8_E5M2()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x2_e5m2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x4_e5m2> = 4;
#  endif // _CCCL_HAS_NVFP8_E5M2()

#  if _CCCL_HAS_NVFP8_E8M0()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x2_e8m0> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp8x4_e8m0> = 4;
#  endif // _CCCL_HAS_NVFP8_E8M0()

#  if _CCCL_HAS_NVFP6_E2M3()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp6x2_e2m3> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp6x4_e2m3> = 4;
#  endif // _CCCL_HAS_NVFP6_E2M3()

#  if _CCCL_HAS_NVFP6_E3M2()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp6x2_e3m2> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp6x4_e3m2> = 4;
#  endif // _CCCL_HAS_NVFP6_E3M2()

#  if _CCCL_HAS_NVFP4_E2M1()
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp4x2_e2m1> = 2;
template <>
inline constexpr ::cuda::std::size_t vector_size_v<::__nv_fp4x4_e2m1> = 4;
#  endif // _CCCL_HAS_NVFP4_E2M1()

template <class _Tp>
using vector_size = ::cuda::std::integral_constant<::cuda::std::size_t, vector_size_v<_Tp>>;

_CCCL_SUPPRESS_DEPRECATED_POP
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA___TYPE_TRAITS_VECTOR_SIZE_H
