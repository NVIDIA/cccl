//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__TYPE_TRAITS_VECTOR_TYPE_H
#define _CUDA__TYPE_TRAITS_VECTOR_TYPE_H

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
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// vector_type

template <class _Tp, ::cuda::std::size_t _Size>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type
{
  using type _CCCL_NODEBUG = void;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 1>
{
  using type _CCCL_NODEBUG = ::char1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 2>
{
  using type _CCCL_NODEBUG = ::char2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 3>
{
  using type _CCCL_NODEBUG = ::char3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<signed char, 4>
{
  using type _CCCL_NODEBUG = ::char4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 1>
{
  using type _CCCL_NODEBUG = ::uchar1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 2>
{
  using type _CCCL_NODEBUG = ::uchar2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 3>
{
  using type _CCCL_NODEBUG = ::uchar3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned char, 4>
{
  using type _CCCL_NODEBUG = ::uchar4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 1>
{
  using type _CCCL_NODEBUG = ::short1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 2>
{
  using type _CCCL_NODEBUG = ::short2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 3>
{
  using type _CCCL_NODEBUG = ::short3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<short, 4>
{
  using type _CCCL_NODEBUG = ::short4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 1>
{
  using type _CCCL_NODEBUG = ::ushort1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 2>
{
  using type _CCCL_NODEBUG = ::ushort2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 3>
{
  using type _CCCL_NODEBUG = ::ushort3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned short, 4>
{
  using type _CCCL_NODEBUG = ::ushort4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 1>
{
  using type _CCCL_NODEBUG = ::int1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 2>
{
  using type _CCCL_NODEBUG = ::int2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 3>
{
  using type _CCCL_NODEBUG = ::int3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<int, 4>
{
  using type _CCCL_NODEBUG = ::int4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 1>
{
  using type _CCCL_NODEBUG = ::uint1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 2>
{
  using type _CCCL_NODEBUG = ::uint2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 3>
{
  using type _CCCL_NODEBUG = ::uint3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned, 4>
{
  using type _CCCL_NODEBUG = ::uint4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 1>
{
  using type _CCCL_NODEBUG = ::long1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 2>
{
  using type _CCCL_NODEBUG = ::long2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 3>
{
  using type _CCCL_NODEBUG = ::long3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::long4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type _CCCL_NODEBUG = ::long4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 1>
{
  using type _CCCL_NODEBUG = ::ulong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 2>
{
  using type _CCCL_NODEBUG = ::ulong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 3>
{
  using type _CCCL_NODEBUG = ::ulong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::ulong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type _CCCL_NODEBUG = ::ulong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 1>
{
  using type _CCCL_NODEBUG = ::longlong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 2>
{
  using type _CCCL_NODEBUG = ::longlong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 3>
{
  using type _CCCL_NODEBUG = ::longlong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<long long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::longlong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type _CCCL_NODEBUG = ::longlong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 1>
{
  using type _CCCL_NODEBUG = ::ulonglong1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 2>
{
  using type _CCCL_NODEBUG = ::ulonglong2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 3>
{
  using type _CCCL_NODEBUG = ::ulonglong3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<unsigned long long, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::ulonglong4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type _CCCL_NODEBUG = ::ulonglong4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 1>
{
  using type _CCCL_NODEBUG = ::float1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 2>
{
  using type _CCCL_NODEBUG = ::float2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 3>
{
  using type _CCCL_NODEBUG = ::float3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<float, 4>
{
  using type _CCCL_NODEBUG = ::float4;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 1>
{
  using type _CCCL_NODEBUG = ::double1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 2>
{
  using type _CCCL_NODEBUG = ::double2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 3>
{
  using type _CCCL_NODEBUG = ::double3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<double, 4>
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
  using type = ::double4_32a;
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  using type _CCCL_NODEBUG = ::double4;
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
};

#  if _CCCL_HAS_NVFP16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__half, 2>
{
  using type _CCCL_NODEBUG = ::__half2;
};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_bfloat16, 2>
{
  using type _CCCL_NODEBUG = ::__nv_bfloat162;
};
#  endif // _CCCL_HAS_NVBF16()

#  if _CCCL_HAS_NVFP8_E4M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e4m3, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x2_e4m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e4m3, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x4_e4m3;
};
#  endif // _CCCL_HAS_NVFP8_E4M3()

#  if _CCCL_HAS_NVFP8_E5M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e5m2, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x2_e5m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e5m2, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x4_e5m2;
};
#  endif // _CCCL_HAS_NVFP8_E5M2()

#  if _CCCL_HAS_NVFP8_E8M0()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e8m0, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x2_e8m0;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp8_e8m0, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp8x4_e8m0;
};
#  endif // _CCCL_HAS_NVFP8_E8M0()

#  if _CCCL_HAS_NVFP6_E2M3()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e2m3, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp6x2_e2m3;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e2m3, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp6x4_e2m3;
};
#  endif // _CCCL_HAS_NVFP6_E2M3()

#  if _CCCL_HAS_NVFP6_E3M2()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e3m2, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp6x2_e3m2;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp6_e3m2, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp6x4_e3m2;
};
#  endif // _CCCL_HAS_NVFP6_E3M2()

#  if _CCCL_HAS_NVFP4_E2M1()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp4_e2m1, 2>
{
  using type _CCCL_NODEBUG = ::__nv_fp4x2_e2m1;
};
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT vector_type<::__nv_fp4_e2m1, 4>
{
  using type _CCCL_NODEBUG = ::__nv_fp4x4_e2m1;
};
#  endif // _CCCL_HAS_NVFP4_E2M1()

template <class _Tp, ::cuda::std::size_t _Size>
using vector_type_t = typename vector_type<_Tp, _Size>::type;

// has_vector_type

template <class _Tp, ::cuda::std::size_t _Size>
inline constexpr bool has_vector_type_v = !::cuda::std::is_same_v<vector_type_t<_Tp, _Size>, void>;

template <class _Tp, ::cuda::std::size_t _Size>
using has_vector_type = ::cuda::std::bool_constant<has_vector_type_v<_Tp, _Size>>;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA__TYPE_TRAITS_VECTOR_TYPE_H
