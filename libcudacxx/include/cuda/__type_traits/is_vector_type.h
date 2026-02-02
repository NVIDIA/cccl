//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
#define _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  if !_CCCL_CUDA_COMPILATION()
#    include <vector_types.h>
#  endif // !_CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class _Tp>
inline constexpr bool is_vector_type_v = false;

template <>
inline constexpr bool is_vector_type_v<::char1> = true;
template <>
inline constexpr bool is_vector_type_v<::char2> = true;
template <>
inline constexpr bool is_vector_type_v<::char3> = true;
template <>
inline constexpr bool is_vector_type_v<::char4> = true;

template <>
inline constexpr bool is_vector_type_v<::uchar1> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar2> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar3> = true;
template <>
inline constexpr bool is_vector_type_v<::uchar4> = true;

template <>
inline constexpr bool is_vector_type_v<::short1> = true;
template <>
inline constexpr bool is_vector_type_v<::short2> = true;
template <>
inline constexpr bool is_vector_type_v<::short3> = true;
template <>
inline constexpr bool is_vector_type_v<::short4> = true;

template <>
inline constexpr bool is_vector_type_v<::ushort1> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort2> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort3> = true;
template <>
inline constexpr bool is_vector_type_v<::ushort4> = true;

template <>
inline constexpr bool is_vector_type_v<::int1> = true;
template <>
inline constexpr bool is_vector_type_v<::int2> = true;
template <>
inline constexpr bool is_vector_type_v<::int3> = true;
template <>
inline constexpr bool is_vector_type_v<::int4> = true;

template <>
inline constexpr bool is_vector_type_v<::uint1> = true;
template <>
inline constexpr bool is_vector_type_v<::uint2> = true;
template <>
inline constexpr bool is_vector_type_v<::uint3> = true;
template <>
inline constexpr bool is_vector_type_v<::uint4> = true;

template <>
inline constexpr bool is_vector_type_v<::long1> = true;
template <>
inline constexpr bool is_vector_type_v<::long2> = true;
template <>
inline constexpr bool is_vector_type_v<::long3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::long4_16a> = true;
template <>
inline constexpr bool is_vector_type_v<::long4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::long4> = true;

template <>
inline constexpr bool is_vector_type_v<::ulong1> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong2> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::ulong4_16a> = true;
template <>
inline constexpr bool is_vector_type_v<::ulong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::ulong4> = true;

template <>
inline constexpr bool is_vector_type_v<::longlong1> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong2> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::longlong4_16a> = true;
template <>
inline constexpr bool is_vector_type_v<::longlong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::longlong4> = true;

template <>
inline constexpr bool is_vector_type_v<::ulonglong1> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong2> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::ulonglong4_16a> = true;
template <>
inline constexpr bool is_vector_type_v<::ulonglong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::ulonglong4> = true;

template <>
inline constexpr bool is_vector_type_v<::float1> = true;
template <>
inline constexpr bool is_vector_type_v<::float2> = true;
template <>
inline constexpr bool is_vector_type_v<::float3> = true;
template <>
inline constexpr bool is_vector_type_v<::float4> = true;

template <>
inline constexpr bool is_vector_type_v<::double1> = true;
template <>
inline constexpr bool is_vector_type_v<::double2> = true;
template <>
inline constexpr bool is_vector_type_v<::double3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::double4_16a> = true;
template <>
inline constexpr bool is_vector_type_v<::double4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool is_vector_type_v<::double4> = true;

template <>
inline constexpr bool is_vector_type_v<::dim3> = true;

template <typename _Tp>
inline constexpr bool is_extended_fp_vector_type_v = false;

#  if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__half2> = true;
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_bfloat162> = true;
#  endif // _CCCL_HAS_NVBF16()

#  if _CCCL_HAS_NVFP8()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x2_e4m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x2_e5m2> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x4_e4m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x4_e5m2> = true;
#    if _CCCL_CTK_AT_LEAST(12, 8)
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x2_e8m0> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp8x4_e8m0> = true;
#    endif // _CCCL_CTK_AT_LEAST(12, 8)
#  endif // _CCCL_HAS_NVFP8()

#  if _CCCL_HAS_NVFP6()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp6x2_e2m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp6x2_e3m2> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp6x4_e2m3> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp6x4_e3m2> = true;
#  endif // _CCCL_HAS_NVFP6()

#  if _CCCL_HAS_NVFP4()
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp4x2_e2m1> = true;
template <>
inline constexpr bool is_extended_fp_vector_type_v<::__nv_fp4x4_e2m1> = true;
#  endif // _CCCL_HAS_NVFP4()

_CCCL_SUPPRESS_DEPRECATED_POP
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA__TYPE_TRAITS_IS_VECTOR_TYPE_H
