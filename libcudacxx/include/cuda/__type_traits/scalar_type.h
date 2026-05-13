//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TYPE_TRAITS_SCALAR_TYPE_H
#define _CUDA___TYPE_TRAITS_SCALAR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<const _Tp> : scalar_type<_Tp>
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<volatile _Tp> : scalar_type<_Tp>
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<const volatile _Tp> : scalar_type<_Tp>
{};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::char1>
{
  using type = signed char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::char2>
{
  using type = signed char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::char3>
{
  using type = signed char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::char4>
{
  using type = signed char;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uchar1>
{
  using type = unsigned char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uchar2>
{
  using type = unsigned char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uchar3>
{
  using type = unsigned char;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uchar4>
{
  using type = unsigned char;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::short1>
{
  using type = short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::short2>
{
  using type = short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::short3>
{
  using type = short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::short4>
{
  using type = short;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ushort1>
{
  using type = unsigned short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ushort2>
{
  using type = unsigned short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ushort3>
{
  using type = unsigned short;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ushort4>
{
  using type = unsigned short;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::int1>
{
  using type = int;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::int2>
{
  using type = int;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::int3>
{
  using type = int;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::int4>
{
  using type = int;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uint1>
{
  using type = unsigned;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uint2>
{
  using type = unsigned;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uint3>
{
  using type = unsigned;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::uint4>
{
  using type = unsigned;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long1>
{
  using type = long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long2>
{
  using type = long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long3>
{
  using type = long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long4>
{
  using type = long;
};
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long4_16a>
{
  using type = long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::long4_32a>
{
  using type = long;
};
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong1>
{
  using type = unsigned long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong2>
{
  using type = unsigned long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong3>
{
  using type = unsigned long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong4>
{
  using type = unsigned long;
};
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong4_16a>
{
  using type = unsigned long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulong4_32a>
{
  using type = unsigned long;
};
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong1>
{
  using type = long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong2>
{
  using type = long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong3>
{
  using type = long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong4>
{
  using type = long long;
};
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong4_16a>
{
  using type = long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::longlong4_32a>
{
  using type = long long;
};
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong1>
{
  using type = unsigned long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong2>
{
  using type = unsigned long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong3>
{
  using type = unsigned long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong4>
{
  using type = unsigned long long;
};
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong4_16a>
{
  using type = unsigned long long;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::ulonglong4_32a>
{
  using type = unsigned long long;
};
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::float1>
{
  using type = float;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::float2>
{
  using type = float;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::float3>
{
  using type = float;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::float4>
{
  using type = float;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double1>
{
  using type = double;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double2>
{
  using type = double;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double3>
{
  using type = double;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double4>
{
  using type = double;
};
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double4_16a>
{
  using type = double;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::double4_32a>
{
  using type = double;
};
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::dim3>
{
  using type = unsigned;
};

#  if _CCCL_HAS_NVFP16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__half2>
{
  using type = ::__half;
};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_bfloat162>
{
  using type = ::__nv_bfloat16;
};
#  endif // _CCCL_HAS_NVBF16()

#  if _CCCL_HAS_NVFP8_E4M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x2_e4m3>
{
  using type = ::__nv_fp8_e4m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x4_e4m3>
{
  using type = ::__nv_fp8_e4m3;
};
#  endif // _CCCL_HAS_NVFP8_E4M3()

#  if _CCCL_HAS_NVFP8_E5M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x2_e5m2>
{
  using type = ::__nv_fp8_e5m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x4_e5m2>
{
  using type = ::__nv_fp8_e5m2;
};
#  endif // _CCCL_HAS_NVFP8_E5M2()

#  if _CCCL_HAS_NVFP8_E8M0()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x2_e8m0>
{
  using type = ::__nv_fp8_e8m0;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp8x4_e8m0>
{
  using type = ::__nv_fp8_e8m0;
};
#  endif // _CCCL_HAS_NVFP8_E8M0()

#  if _CCCL_HAS_NVFP6_E2M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp6x2_e2m3>
{
  using type = ::__nv_fp6_e2m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp6x4_e2m3>
{
  using type = ::__nv_fp6_e2m3;
};
#  endif // _CCCL_HAS_NVFP6_E2M3()

#  if _CCCL_HAS_NVFP6_E3M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp6x2_e3m2>
{
  using type = ::__nv_fp6_e3m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp6x4_e3m2>
{
  using type = ::__nv_fp6_e3m2;
};
#  endif // _CCCL_HAS_NVFP6_E3M2()

#  if _CCCL_HAS_NVFP4_E2M1()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp4x2_e2m1>
{
  using type = ::__nv_fp4_e2m1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT scalar_type<::__nv_fp4x4_e2m1>
{
  using type = ::__nv_fp4_e2m1;
};
#  endif // _CCCL_HAS_NVFP4_E2M1()

template <class _Tp>
using scalar_type_t = typename scalar_type<_Tp>::type;

_CCCL_SUPPRESS_DEPRECATED_POP
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA___TYPE_TRAITS_SCALAR_TYPE_H
