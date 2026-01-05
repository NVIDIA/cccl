//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
#  include <cuda/std/__type_traits/is_same.h>

#  if !_CCCL_CUDA_COMPILATION()
#    include <vector_types.h>
#  endif // !_CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, ::cuda::std::size_t _Size>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __cccl_vector_type_t_impl() noexcept
{
  if constexpr (::cuda::std::is_same_v<_Tp, signed char>)
  {
    if constexpr (_Size == 1)
    {
      return ::char1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::char2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::char3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::char4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, unsigned char>)
  {
    if constexpr (_Size == 1)
    {
      return ::uchar1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::uchar2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::uchar3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::uchar4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, short>)
  {
    if constexpr (_Size == 1)
    {
      return ::short1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::short2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::short3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::short4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, unsigned short>)
  {
    if constexpr (_Size == 1)
    {
      return ::ushort1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::ushort2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::ushort3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::ushort4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, int>)
  {
    if constexpr (_Size == 1)
    {
      return ::int1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::int2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::int3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::int4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, unsigned>)
  {
    if constexpr (_Size == 1)
    {
      return ::uint1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::uint2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::uint3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::uint4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, long>)
  {
    if constexpr (_Size == 1)
    {
      return ::long1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::long2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::long3{};
    }
    else if constexpr (_Size == 4)
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return ::long4_32a{};
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
      return ::long4{};
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, unsigned long>)
  {
    if constexpr (_Size == 1)
    {
      return ::ulong1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::ulong2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::ulong3{};
    }
    else if constexpr (_Size == 4)
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return ::ulong4_32a{};
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
      return ::ulong4{};
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, long long>)
  {
    if constexpr (_Size == 1)
    {
      return ::longlong1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::longlong2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::longlong3{};
    }
    else if constexpr (_Size == 4)
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return ::longlong4_32a{};
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
      return ::longlong4{};
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, unsigned long long>)
  {
    if constexpr (_Size == 1)
    {
      return ::ulonglong1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::ulonglong2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::ulonglong3{};
    }
    else if constexpr (_Size == 4)
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return ::ulonglong4_32a{};
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
      return ::ulonglong4{};
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, float>)
  {
    if constexpr (_Size == 1)
    {
      return ::float1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::float2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::float3{};
    }
    else if constexpr (_Size == 4)
    {
      return ::float4{};
    }
    else
    {
      return;
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, double>)
  {
    if constexpr (_Size == 1)
    {
      return ::double1{};
    }
    else if constexpr (_Size == 2)
    {
      return ::double2{};
    }
    else if constexpr (_Size == 3)
    {
      return ::double3{};
    }
    else if constexpr (_Size == 4)
    {
#  if _CCCL_CTK_AT_LEAST(13, 0)
      return ::double4_32a{};
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
      return ::double4{};
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
    }
    else
    {
      return;
    }
  }
  else
  {
    return;
  }
}

template <class _Tp, ::cuda::std::size_t _Size>
using __vector_type_t = decltype(::cuda::__cccl_vector_type_t_impl<_Tp, _Size>());

template <class _Tp, ::cuda::std::size_t _Size>
inline constexpr bool __has_vector_type_v = !::cuda::std::is_same_v<__vector_type_t<_Tp, _Size>, void>;

template <class _Tp>
inline constexpr bool __is_vector_type_v = false;

template <>
inline constexpr bool __is_vector_type_v<::char1> = true;
template <>
inline constexpr bool __is_vector_type_v<::char2> = true;
template <>
inline constexpr bool __is_vector_type_v<::char3> = true;
template <>
inline constexpr bool __is_vector_type_v<::char4> = true;

template <>
inline constexpr bool __is_vector_type_v<::uchar1> = true;
template <>
inline constexpr bool __is_vector_type_v<::uchar2> = true;
template <>
inline constexpr bool __is_vector_type_v<::uchar3> = true;
template <>
inline constexpr bool __is_vector_type_v<::uchar4> = true;

template <>
inline constexpr bool __is_vector_type_v<::short1> = true;
template <>
inline constexpr bool __is_vector_type_v<::short2> = true;
template <>
inline constexpr bool __is_vector_type_v<::short3> = true;
template <>
inline constexpr bool __is_vector_type_v<::short4> = true;

template <>
inline constexpr bool __is_vector_type_v<::ushort1> = true;
template <>
inline constexpr bool __is_vector_type_v<::ushort2> = true;
template <>
inline constexpr bool __is_vector_type_v<::ushort3> = true;
template <>
inline constexpr bool __is_vector_type_v<::ushort4> = true;

template <>
inline constexpr bool __is_vector_type_v<::int1> = true;
template <>
inline constexpr bool __is_vector_type_v<::int2> = true;
template <>
inline constexpr bool __is_vector_type_v<::int3> = true;
template <>
inline constexpr bool __is_vector_type_v<::int4> = true;

template <>
inline constexpr bool __is_vector_type_v<::uint1> = true;
template <>
inline constexpr bool __is_vector_type_v<::uint2> = true;
template <>
inline constexpr bool __is_vector_type_v<::uint3> = true;
template <>
inline constexpr bool __is_vector_type_v<::uint4> = true;

template <>
inline constexpr bool __is_vector_type_v<::long1> = true;
template <>
inline constexpr bool __is_vector_type_v<::long2> = true;
template <>
inline constexpr bool __is_vector_type_v<::long3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::long4_16a> = true;
template <>
inline constexpr bool __is_vector_type_v<::long4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::long4> = true;

template <>
inline constexpr bool __is_vector_type_v<::ulong1> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulong2> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::ulong4_16a> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::ulong4> = true;

template <>
inline constexpr bool __is_vector_type_v<::longlong1> = true;
template <>
inline constexpr bool __is_vector_type_v<::longlong2> = true;
template <>
inline constexpr bool __is_vector_type_v<::longlong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::longlong4_16a> = true;
template <>
inline constexpr bool __is_vector_type_v<::longlong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::longlong4> = true;

template <>
inline constexpr bool __is_vector_type_v<::ulonglong1> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulonglong2> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulonglong3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::ulonglong4_16a> = true;
template <>
inline constexpr bool __is_vector_type_v<::ulonglong4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::ulonglong4> = true;

template <>
inline constexpr bool __is_vector_type_v<::float1> = true;
template <>
inline constexpr bool __is_vector_type_v<::float2> = true;
template <>
inline constexpr bool __is_vector_type_v<::float3> = true;
template <>
inline constexpr bool __is_vector_type_v<::float4> = true;

template <>
inline constexpr bool __is_vector_type_v<::double1> = true;
template <>
inline constexpr bool __is_vector_type_v<::double2> = true;
template <>
inline constexpr bool __is_vector_type_v<::double3> = true;
#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::double4_16a> = true;
template <>
inline constexpr bool __is_vector_type_v<::double4_32a> = true;
#  endif // ^^^ _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline constexpr bool __is_vector_type_v<::double4> = true;

template <>
inline constexpr bool __is_vector_type_v<::dim3> = true;

template <typename _Tp>
inline constexpr bool __is_extended_fp_vector_type_v = false;

#  if _CCCL_HAS_NVFP8()
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_bfloat162> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__half2> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x2_e4m3> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x2_e5m2> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x4_e4m3> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x4_e5m2> = true;
#    if _CCCL_CTK_AT_LEAST(12, 8)
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x2_e8m0> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp8x4_e8m0> = true;
#    endif // _CCCL_CTK_AT_LEAST(12, 8)
#  endif // _CCCL_HAS_NVFP8()

#  if _CCCL_HAS_NVFP6()
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp6x2_e2m3> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp6x2_e3m2> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp6x4_e2m3> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp6x4_e3m2> = true;
#  endif // _CCCL_HAS_NVFP6()

#  if _CCCL_HAS_NVFP4()
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp4x2_e2m1> = true;
template <>
inline constexpr bool __is_extended_fp_vector_type_v<::__nv_fp4x4_e2m1> = true;
#  endif // _CCCL_HAS_NVFP4()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_CTK()
#endif // _CUDA__TYPE_TRAITS_VECTOR_TYPE_H
