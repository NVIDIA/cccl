//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// automatic vectorization for float2 is not supported (until CUDA 13.2)
// TODO(fbusato): extend for other GPU archs in the future
// TODO(fbusato): check 5361571, remove this path once the feature is supported
#if _CCCL_HAS_SIMD_F32X2()

#  include <cuda/std/__simd/specializations/fixed_size_storage.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

[[nodiscard]] _CCCL_DEVICE_API inline ::float2 __add_f32x2(const ::float2 __lhs, const ::float2 __rhs) noexcept
{
  ::float2 __result{};
#  if _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (__result = ::__fadd2_rn(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_F32X2_PTX() // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "add.f32x2 __result, __lhs, __rhs;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result.x), "=f"(__result.y)
      : "f"(__lhs.x), "f"(__lhs.y), "f"(__rhs.x), "f"(__rhs.y));
#  endif // _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  return __result;
}

[[nodiscard]] _CCCL_DEVICE_API inline ::float2 __mul_f32x2(const ::float2 __lhs, const ::float2 __rhs) noexcept
{
  ::float2 __result{};
#  if _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (__result = ::__fmul2_rn(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_F32X2_PTX() // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "mul.f32x2 __result, __lhs, __rhs;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result.x), "=f"(__result.y)
      : "f"(__lhs.x), "f"(__lhs.y), "f"(__rhs.x), "f"(__rhs.y));
#  endif // _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  return __result;
}

[[nodiscard]] _CCCL_DEVICE_API inline ::float2 __sub_f32x2(const ::float2 __lhs, const ::float2 __rhs) noexcept
{
  ::float2 __result{};
#  if _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (__result = ::__fadd2_rn(__lhs, ::float2{-__rhs.x, -__rhs.y});))
#  elif _CCCL_HAS_SIMD_F32X2_PTX() // PTX ISA 8.6
  NV_IF_TARGET(
    NV_IS_EXACTLY_SM_100,
    (asm("{.reg .b64 __lhs, __rhs, __result;"
         "mov.b64 __lhs, {%2, %3};"
         "mov.b64 __rhs, {%4, %5};"
         "sub.f32x2 __result, __lhs, __rhs;"
         "mov.b64 {%0, %1}, __result;}" : "=f"(__result.x),
         "=f"(__result.y) : "f"(__lhs.x),
         "f"(__lhs.y),
         "f"(__rhs.x),
         "f"(__rhs.y));))
#  endif // _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  return __result;
}

[[nodiscard]] _CCCL_DEVICE_API inline ::float2
__fma_f32x2(const ::float2 __lhs, const ::float2 __rhs, const ::float2 __add) noexcept
{
  ::float2 __result{};
#  if _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (__result = ::__ffma2_rn(__lhs, __rhs, __add);))
#  elif _CCCL_HAS_SIMD_F32X2_PTX() // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __add, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "mov.b64 __add, {%6, %7};"
      "fma.rn.f32x2 __result, __lhs, __rhs, __add;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result.x), "=f"(__result.y)
      : "f"(__lhs.x), "f"(__lhs.y), "f"(__rhs.x), "f"(__rhs.y), "f"(__add.x), "f"(__add.y));
#  endif // _CCCL_HAS_SIMD_F32X2_INTRINSICS()
  return __result;
}

template <__simd_size_type _Np>
using __simd_storage_f32 = __simd_storage<float, __fixed_size<_Np>>;

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__plus_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    const auto __lhs_value   = ::float2{__lhs.__data[__i], __lhs.__data[__i + 1]};
    const auto __rhs_value   = ::float2{__rhs.__data[__i], __rhs.__data[__i + 1]};
    const auto __value       = ::cuda::std::simd::__add_f32x2(__lhs_value, __rhs_value);
    __result.__data[__i]     = __value.x;
    __result.__data[__i + 1] = __value.y;
  }
  if constexpr (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] + __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__minus_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    const auto __lhs_value   = ::float2{__lhs.__data[__i], __lhs.__data[__i + 1]};
    const auto __rhs_value   = ::float2{__rhs.__data[__i], __rhs.__data[__i + 1]};
    const auto __value       = ::cuda::std::simd::__sub_f32x2(__lhs_value, __rhs_value);
    __result.__data[__i]     = __value.x;
    __result.__data[__i + 1] = __value.y;
  }
  if constexpr (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] - __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__multiplies_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    const auto __lhs_value   = ::float2{__lhs.__data[__i], __lhs.__data[__i + 1]};
    const auto __rhs_value   = ::float2{__rhs.__data[__i], __rhs.__data[__i + 1]};
    const auto __value       = ::cuda::std::simd::__mul_f32x2(__lhs_value, __rhs_value);
    __result.__data[__i]     = __value.x;
    __result.__data[__i + 1] = __value.y;
  }
  if constexpr (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] * __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__fma_f32x2(const __simd_storage_f32<_Np>& __lhs,
            const __simd_storage_f32<_Np>& __rhs,
            const __simd_storage_f32<_Np>& __add) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    const auto __lhs_value   = ::float2{__lhs.__data[__i], __lhs.__data[__i + 1]};
    const auto __rhs_value   = ::float2{__rhs.__data[__i], __rhs.__data[__i + 1]};
    const auto __add_value   = ::float2{__add.__data[__i], __add.__data[__i + 1]};
    const auto __value       = ::cuda::std::simd::__fma_f32x2(__lhs_value, __rhs_value, __add_value);
    __result.__data[__i]     = __value.x;
    __result.__data[__i + 1] = __value.y;
  }
  if constexpr (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] * __rhs.__data[_Np - 1] + __add.__data[_Np - 1];
  }
  return __result;
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_F32X2()
#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H
