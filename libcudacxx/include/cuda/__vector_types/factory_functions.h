//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___VECTOR_TYPES_FACTORY_FUNCTIONS_H
#define _CUDA___VECTOR_TYPES_FACTORY_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__vector_types/traits.h>
#  include <cuda/__vector_types/types.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/type_identity.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct value_broadcast_t
{
  _CCCL_HIDE_FROM_ABI explicit constexpr value_broadcast_t() noexcept = default;
};

_CCCL_GLOBAL_CONSTANT value_broadcast_t value_broadcast{};

// make_charN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char1 make_char1() noexcept
{
  return char1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char1 make_char1(signed char __x) noexcept
{
  return char1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char1 make_char1(value_broadcast_t, signed char __x) noexcept
{
  return char1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char2 make_char2() noexcept
{
  return char2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char2 make_char2(signed char __x, signed char __y) noexcept
{
  return char2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char2 make_char2(value_broadcast_t, signed char __x) noexcept
{
  return char2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char3 make_char3() noexcept
{
  return char3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char3
make_char3(signed char __x, signed char __y, signed char __z) noexcept
{
  return char3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char3 make_char3(value_broadcast_t, signed char __x) noexcept
{
  return char3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char4 make_char4() noexcept
{
  return char4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char4
make_char4(signed char __x, signed char __y, signed char __z, signed char __w) noexcept
{
  return char4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr char4 make_char4(value_broadcast_t, signed char __x) noexcept
{
  return char4{__x, __x, __x, __x};
}

// make_ucharN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar1 make_uchar1() noexcept
{
  return uchar1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar1 make_uchar1(unsigned char __x) noexcept
{
  return uchar1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar1 make_uchar1(value_broadcast_t, unsigned char __x) noexcept
{
  return uchar1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar2 make_uchar2() noexcept
{
  return uchar2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar2 make_uchar2(unsigned char __x, unsigned char __y) noexcept
{
  return uchar2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar2 make_uchar2(value_broadcast_t, unsigned char __x) noexcept
{
  return uchar2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar3 make_uchar3() noexcept
{
  return uchar3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar3
make_uchar3(unsigned char __x, unsigned char __y, unsigned char __z) noexcept
{
  return uchar3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar3 make_uchar3(value_broadcast_t, unsigned char __x) noexcept
{
  return uchar3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar4 make_uchar4() noexcept
{
  return uchar4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar4
make_uchar4(unsigned char __x, unsigned char __y, unsigned char __z, unsigned char __w) noexcept
{
  return uchar4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uchar4 make_uchar4(value_broadcast_t, unsigned char __x) noexcept
{
  return uchar4{__x, __x, __x, __x};
}

// make_shorN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short1 make_short1() noexcept
{
  return short1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short1 make_short1(short __x) noexcept
{
  return short1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short1 make_short1(value_broadcast_t, short __x) noexcept
{
  return short1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short2 make_short2() noexcept
{
  return short2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short2 make_short2(short __x, short __y) noexcept
{
  return short2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short2 make_short2(value_broadcast_t, short __x) noexcept
{
  return short2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short3 make_short3() noexcept
{
  return short3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short3 make_short3(short __x, short __y, short __z) noexcept
{
  return short3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short3 make_short3(value_broadcast_t, short __x) noexcept
{
  return short3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short4 make_short4() noexcept
{
  return short4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short4 make_short4(short __x, short __y, short __z, short __w) noexcept
{
  return short4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr short4 make_short4(value_broadcast_t, short __x) noexcept
{
  return short4{__x, __x, __x, __x};
}

// make_ushortN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort1 make_ushort1() noexcept
{
  return ushort1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort1 make_ushort1(unsigned short __x) noexcept
{
  return ushort1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort1 make_ushort1(value_broadcast_t, unsigned short __x) noexcept
{
  return ushort1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort2 make_ushort2() noexcept
{
  return ushort2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort2 make_ushort2(unsigned short __x, unsigned short __y) noexcept
{
  return ushort2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort2 make_ushort2(value_broadcast_t, unsigned short __x) noexcept
{
  return ushort2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort3 make_ushort3() noexcept
{
  return ushort3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort3
make_ushort3(unsigned short __x, unsigned short __y, unsigned short __z) noexcept
{
  return ushort3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort3 make_ushort3(value_broadcast_t, unsigned short __x) noexcept
{
  return ushort3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort4 make_ushort4() noexcept
{
  return ushort4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort4
make_ushort4(unsigned short __x, unsigned short __y, unsigned short __z, unsigned short __w) noexcept
{
  return ushort4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ushort4 make_ushort4(value_broadcast_t, unsigned short __x) noexcept
{
  return ushort4{__x, __x, __x, __x};
}

// make_intN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int1 make_int1() noexcept
{
  return int1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int1 make_int1(int __x) noexcept
{
  return int1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int1 make_int1(value_broadcast_t, int __x) noexcept
{
  return int1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int2 make_int2() noexcept
{
  return int2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int2 make_int2(int __x, int __y) noexcept
{
  return int2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int2 make_int2(value_broadcast_t, int __x) noexcept
{
  return int2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int3 make_int3() noexcept
{
  return int3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int3 make_int3(int __x, int __y, int __z) noexcept
{
  return int3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int3 make_int3(value_broadcast_t, int __x) noexcept
{
  return int3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int4 make_int4() noexcept
{
  return int4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int4 make_int4(int __x, int __y, int __z, int __w) noexcept
{
  return int4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int4 make_int4(value_broadcast_t, int __x) noexcept
{
  return int4{__x, __x, __x, __x};
}

// make_uintN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint1 make_uint1() noexcept
{
  return uint1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint1 make_uint1(unsigned int __x) noexcept
{
  return uint1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint1 make_uint1(value_broadcast_t, unsigned int __x) noexcept
{
  return uint1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint2 make_uint2() noexcept
{
  return uint2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint2 make_uint2(unsigned int __x, unsigned int __y) noexcept
{
  return uint2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint2 make_uint2(value_broadcast_t, unsigned int __x) noexcept
{
  return uint2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint3 make_uint3() noexcept
{
  return uint3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint3
make_uint3(unsigned int __x, unsigned int __y, unsigned int __z) noexcept
{
  return uint3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint3 make_uint3(value_broadcast_t, unsigned int __x) noexcept
{
  return uint3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint4 make_uint4() noexcept
{
  return uint4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint4
make_uint4(unsigned int __x, unsigned int __y, unsigned int __z, unsigned int __w) noexcept
{
  return uint4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint4 make_uint4(value_broadcast_t, unsigned int __x) noexcept
{
  return uint4{__x, __x, __x, __x};
}

// make_longN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long1 make_long1() noexcept
{
  return long1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long1 make_long1(long __x) noexcept
{
  return long1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long1 make_long1(value_broadcast_t, long __x) noexcept
{
  return long1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long2 make_long2() noexcept
{
  return long2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long2 make_long2(long __x, long __y) noexcept
{
  return long2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long2 make_long2(value_broadcast_t, long __x) noexcept
{
  return long2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long3 make_long3() noexcept
{
  return long3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long3 make_long3(long __x, long __y, long __z) noexcept
{
  return long3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long3 make_long3(value_broadcast_t, long __x) noexcept
{
  return long3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long4 make_long4() noexcept
{
  return long4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long4 make_long4(long __x, long __y, long __z, long __w) noexcept
{
  return long4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr long4 make_long4(value_broadcast_t, long __x) noexcept
{
  return long4{__x, __x, __x, __x};
}

// make_ulongN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong1 make_ulong1() noexcept
{
  return ulong1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong1 make_ulong1(unsigned long __x) noexcept
{
  return ulong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong1 make_ulong1(value_broadcast_t, unsigned long __x) noexcept
{
  return ulong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong2 make_ulong2() noexcept
{
  return ulong2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong2 make_ulong2(unsigned long __x, unsigned long __y) noexcept
{
  return ulong2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong2 make_ulong2(value_broadcast_t, unsigned long __x) noexcept
{
  return ulong2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong3 make_ulong3() noexcept
{
  return ulong3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong3
make_ulong3(unsigned long __x, unsigned long __y, unsigned long __z) noexcept
{
  return ulong3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong3 make_ulong3(value_broadcast_t, unsigned long __x) noexcept
{
  return ulong3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong4 make_ulong4() noexcept
{
  return ulong4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong4
make_ulong4(unsigned long __x, unsigned long __y, unsigned long __z, unsigned long __w) noexcept
{
  return ulong4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulong4 make_ulong4(value_broadcast_t, unsigned long __x) noexcept
{
  return ulong4{__x, __x, __x, __x};
}

// make_longlongN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong1 make_longlong1() noexcept
{
  return longlong1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong1 make_longlong1(long long __x) noexcept
{
  return longlong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong1 make_longlong1(value_broadcast_t, long long __x) noexcept
{
  return longlong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong2 make_longlong2() noexcept
{
  return longlong2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong2 make_longlong2(long long __x, long long __y) noexcept
{
  return longlong2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong2 make_longlong2(value_broadcast_t, long long __x) noexcept
{
  return longlong2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong3 make_longlong3() noexcept
{
  return longlong3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong3
make_longlong3(long long __x, long long __y, long long __z) noexcept
{
  return longlong3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong3 make_longlong3(value_broadcast_t, long long __x) noexcept
{
  return longlong3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong4 make_longlong4() noexcept
{
  return longlong4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong4
make_longlong4(long long __x, long long __y, long long __z, long long __w) noexcept
{
  return longlong4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr longlong4 make_longlong4(value_broadcast_t, long long __x) noexcept
{
  return longlong4{__x, __x, __x, __x};
}

// make_ulonglongN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong1 make_ulonglong1() noexcept
{
  return ulonglong1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong1 make_ulonglong1(unsigned long long __x) noexcept
{
  return ulonglong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong1
make_ulonglong1(value_broadcast_t, unsigned long long __x) noexcept
{
  return ulonglong1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong2 make_ulonglong2() noexcept
{
  return ulonglong2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong2
make_ulonglong2(unsigned long long __x, unsigned long long __y) noexcept
{
  return ulonglong2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong2
make_ulonglong2(value_broadcast_t, unsigned long long __x) noexcept
{
  return ulonglong2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong3 make_ulonglong3() noexcept
{
  return ulonglong3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong3
make_ulonglong3(unsigned long long __x, unsigned long long __y, unsigned long long __z) noexcept
{
  return ulonglong3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong3
make_ulonglong3(value_broadcast_t, unsigned long long __x) noexcept
{
  return ulonglong3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong4 make_ulonglong4() noexcept
{
  return ulonglong4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong4
make_ulonglong4(unsigned long long __x, unsigned long long __y, unsigned long long __z, unsigned long long __w) noexcept
{
  return ulonglong4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr ulonglong4
make_ulonglong4(value_broadcast_t, unsigned long long __x) noexcept
{
  return ulonglong4{__x, __x, __x, __x};
}

// make_floatN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float1 make_float1() noexcept
{
  return float1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float1 make_float1(float __x) noexcept
{
  return float1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float1 make_float1(value_broadcast_t, float __x) noexcept
{
  return float1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float2 make_float2() noexcept
{
  return float2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float2 make_float2(float __x, float __y) noexcept
{
  return float2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float2 make_float2(value_broadcast_t, float __x) noexcept
{
  return float2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float3 make_float3() noexcept
{
  return float3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float3 make_float3(float __x, float __y, float __z) noexcept
{
  return float3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float3 make_float3(value_broadcast_t, float __x) noexcept
{
  return float3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float4 make_float4() noexcept
{
  return float4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float4 make_float4(float __x, float __y, float __z, float __w) noexcept
{
  return float4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr float4 make_float4(value_broadcast_t, float __x) noexcept
{
  return float4{__x, __x, __x, __x};
}

// make_doubleN

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double1 make_double1() noexcept
{
  return double1{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double1 make_double1(double __x) noexcept
{
  return double1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double1 make_double1(value_broadcast_t, double __x) noexcept
{
  return double1{__x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double2 make_double2() noexcept
{
  return double2{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double2 make_double2(double __x, double __y) noexcept
{
  return double2{__x, __y};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double2 make_double2(value_broadcast_t, double __x) noexcept
{
  return double2{__x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double3 make_double3() noexcept
{
  return double3{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double3 make_double3(double __x, double __y, double __z) noexcept
{
  return double3{__x, __y, __z};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double3 make_double3(value_broadcast_t, double __x) noexcept
{
  return double3{__x, __x, __x};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double4 make_double4() noexcept
{
  return double4{};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double4
make_double4(double __x, double __y, double __z, double __w) noexcept
{
  return double4{__x, __y, __z, __w};
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr double4 make_double4(value_broadcast_t, double __x) noexcept
{
  return double4{__x, __x, __x, __x};
}

// make_vector

template <class _Tp, size_t _Size>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr vector_type_t<_Tp, _Size> make_vector() noexcept
{
  return vector_type_t<_Tp, _Size>{};
}

template <class _Tp, size_t _Size, class... _Args>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr vector_type_t<_Tp, _Size> make_vector(_Args... __args) noexcept
{
  static_assert(sizeof...(_Args) == _Size, "number of arguments must match vector size");
  return vector_type_t<_Tp, _Size>{__args...};
}

template <class _Tp, size_t _Size>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr vector_type_t<_Tp, _Size>
make_vector(value_broadcast_t, _CUDA_VSTD::type_identity_t<_Tp> __x) noexcept
{
  if constexpr (_Size == 1)
  {
    return vector_type_t<_Tp, _Size>{__x};
  }
  else if constexpr (_Size == 2)
  {
    return vector_type_t<_Tp, _Size>{__x, __x};
  }
  else if constexpr (_Size == 3)
  {
    return vector_type_t<_Tp, _Size>{__x, __x, __x};
  }
  else if constexpr (_Size == 4)
  {
    return vector_type_t<_Tp, _Size>{__x, __x, __x, __x};
  }
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Tp>, "unsupported vector size");
    _CCCL_UNREACHABLE();
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___VECTOR_TYPES_FACTORY_FUNCTIONS_H
