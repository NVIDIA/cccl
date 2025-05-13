//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___VECTOR_TYPES_TRAITS_H
#define _CUDA___VECTOR_TYPES_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__vector_types/types.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/remove_cv.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEVAL bool __cccl_is_valid_vector_size(size_t __size) noexcept
{
  return __size > 0 && __size <= 4;
}

template <class _Tp, size_t _Size>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __cccl_vector_type_t_impl() noexcept
{
  static_assert(::cuda::__cccl_is_valid_vector_size(_Size), "vector size out of range");

  using _Up = _CUDA_VSTD::remove_cv_t<_Tp>;

  if constexpr (_CUDA_VSTD::is_same_v<_Up, signed char>)
  {
    if constexpr (_Size == 1)
    {
      return char1{};
    }
    else if constexpr (_Size == 2)
    {
      return char2{};
    }
    else if constexpr (_Size == 3)
    {
      return char3{};
    }
    else
    {
      return char4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, unsigned char>)
  {
    if constexpr (_Size == 1)
    {
      return uchar1{};
    }
    else if constexpr (_Size == 2)
    {
      return uchar2{};
    }
    else if constexpr (_Size == 3)
    {
      return uchar3{};
    }
    else
    {
      return uchar4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, short>)
  {
    if constexpr (_Size == 1)
    {
      return short1{};
    }
    else if constexpr (_Size == 2)
    {
      return short2{};
    }
    else if constexpr (_Size == 3)
    {
      return short3{};
    }
    else
    {
      return short4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, unsigned short>)
  {
    if constexpr (_Size == 1)
    {
      return ushort1{};
    }
    else if constexpr (_Size == 2)
    {
      return ushort2{};
    }
    else if constexpr (_Size == 3)
    {
      return ushort3{};
    }
    else
    {
      return ushort4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, int>)
  {
    if constexpr (_Size == 1)
    {
      return int1{};
    }
    else if constexpr (_Size == 2)
    {
      return int2{};
    }
    else if constexpr (_Size == 3)
    {
      return int3{};
    }
    else
    {
      return int4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, unsigned int>)
  {
    if constexpr (_Size == 1)
    {
      return uint1{};
    }
    else if constexpr (_Size == 2)
    {
      return uint2{};
    }
    else if constexpr (_Size == 3)
    {
      return uint3{};
    }
    else
    {
      return uint4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, long>)
  {
    if constexpr (_Size == 1)
    {
      return long1{};
    }
    else if constexpr (_Size == 2)
    {
      return long2{};
    }
    else if constexpr (_Size == 3)
    {
      return long3{};
    }
    else
    {
      return long4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, unsigned long>)
  {
    if constexpr (_Size == 1)
    {
      return ulong1{};
    }
    else if constexpr (_Size == 2)
    {
      return ulong2{};
    }
    else if constexpr (_Size == 3)
    {
      return ulong3{};
    }
    else
    {
      return ulong4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, long long>)
  {
    if constexpr (_Size == 1)
    {
      return longlong1{};
    }
    else if constexpr (_Size == 2)
    {
      return longlong2{};
    }
    else if constexpr (_Size == 3)
    {
      return longlong3{};
    }
    else
    {
      return longlong4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, unsigned long long>)
  {
    if constexpr (_Size == 1)
    {
      return ulonglong1{};
    }
    else if constexpr (_Size == 2)
    {
      return ulonglong2{};
    }
    else if constexpr (_Size == 3)
    {
      return ulonglong3{};
    }
    else
    {
      return ulonglong4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, float>)
  {
    if constexpr (_Size == 1)
    {
      return float1{};
    }
    else if constexpr (_Size == 2)
    {
      return float2{};
    }
    else if constexpr (_Size == 3)
    {
      return float3{};
    }
    else
    {
      return float4{};
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, double>)
  {
    if constexpr (_Size == 1)
    {
      return double1{};
    }
    else if constexpr (_Size == 2)
    {
      return double2{};
    }
    else if constexpr (_Size == 3)
    {
      return double3{};
    }
    else
    {
      return double4{};
    }
  }
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Up>, "unsupported vector underlying type");
  }
}

template <class _Tp, size_t _Size>
using vector_type_t = decltype(::cuda::__cccl_vector_type_t_impl<_Tp, _Size>());

template <class _Tp, size_t _Size>
struct vector_type
{
  using type = vector_type_t<_Tp, _Size>;
};

template <class _Tp>
inline constexpr bool is_vector_type_v = false;
template <class _Tp>
inline constexpr bool is_vector_type_v<const _Tp> = is_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_vector_type_v<volatile _Tp> = is_vector_type_v<_Tp>;
template <class _Tp>
inline constexpr bool is_vector_type_v<const volatile _Tp> = is_vector_type_v<_Tp>;

template <>
inline constexpr bool is_vector_type_v<char1> = true;
template <>
inline constexpr bool is_vector_type_v<char2> = true;
template <>
inline constexpr bool is_vector_type_v<char3> = true;
template <>
inline constexpr bool is_vector_type_v<char4> = true;

template <>
inline constexpr bool is_vector_type_v<uchar1> = true;
template <>
inline constexpr bool is_vector_type_v<uchar2> = true;
template <>
inline constexpr bool is_vector_type_v<uchar3> = true;
template <>
inline constexpr bool is_vector_type_v<uchar4> = true;

template <>
inline constexpr bool is_vector_type_v<short1> = true;
template <>
inline constexpr bool is_vector_type_v<short2> = true;
template <>
inline constexpr bool is_vector_type_v<short3> = true;
template <>
inline constexpr bool is_vector_type_v<short4> = true;

template <>
inline constexpr bool is_vector_type_v<ushort1> = true;
template <>
inline constexpr bool is_vector_type_v<ushort2> = true;
template <>
inline constexpr bool is_vector_type_v<ushort3> = true;
template <>
inline constexpr bool is_vector_type_v<ushort4> = true;

template <>
inline constexpr bool is_vector_type_v<int1> = true;
template <>
inline constexpr bool is_vector_type_v<int2> = true;
template <>
inline constexpr bool is_vector_type_v<int3> = true;
template <>
inline constexpr bool is_vector_type_v<int4> = true;

template <>
inline constexpr bool is_vector_type_v<uint1> = true;
template <>
inline constexpr bool is_vector_type_v<uint2> = true;
template <>
inline constexpr bool is_vector_type_v<uint3> = true;
template <>
inline constexpr bool is_vector_type_v<uint4> = true;

template <>
inline constexpr bool is_vector_type_v<long1> = true;
template <>
inline constexpr bool is_vector_type_v<long2> = true;
template <>
inline constexpr bool is_vector_type_v<long3> = true;
template <>
inline constexpr bool is_vector_type_v<long4> = true;

template <>
inline constexpr bool is_vector_type_v<ulong1> = true;
template <>
inline constexpr bool is_vector_type_v<ulong2> = true;
template <>
inline constexpr bool is_vector_type_v<ulong3> = true;
template <>
inline constexpr bool is_vector_type_v<ulong4> = true;

template <>
inline constexpr bool is_vector_type_v<longlong1> = true;
template <>
inline constexpr bool is_vector_type_v<longlong2> = true;
template <>
inline constexpr bool is_vector_type_v<longlong3> = true;
template <>
inline constexpr bool is_vector_type_v<longlong4> = true;

template <>
inline constexpr bool is_vector_type_v<ulonglong1> = true;
template <>
inline constexpr bool is_vector_type_v<ulonglong2> = true;
template <>
inline constexpr bool is_vector_type_v<ulonglong3> = true;
template <>
inline constexpr bool is_vector_type_v<ulonglong4> = true;

template <>
inline constexpr bool is_vector_type_v<float1> = true;
template <>
inline constexpr bool is_vector_type_v<float2> = true;
template <>
inline constexpr bool is_vector_type_v<float3> = true;
template <>
inline constexpr bool is_vector_type_v<float4> = true;

template <>
inline constexpr bool is_vector_type_v<double1> = true;
template <>
inline constexpr bool is_vector_type_v<double2> = true;
template <>
inline constexpr bool is_vector_type_v<double3> = true;
template <>
inline constexpr bool is_vector_type_v<double4> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_vector_type : _CUDA_VSTD::bool_constant<is_vector_type_v<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___VECTOR_TYPES_TRAITS_H
