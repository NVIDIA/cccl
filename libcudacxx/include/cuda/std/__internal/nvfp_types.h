//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_NVFP_TYPES_H
#define _LIBCUDACXX___INTERNAL_NVFP_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

// Prevent resetting of the diagnostic state by guarding the push/pop with a macro
#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_DIAG_PUSH
#  include <cuda_fp16.h>
_CCCL_DIAG_POP
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8()
_CCCL_DIAG_PUSH
#  include <cuda_fp8.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP8()

#if _CCCL_HAS_NVFP6()
_CCCL_DIAG_PUSH
#  include <cuda_fp6.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP6()

#if _CCCL_HAS_NVFP4()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_MSVC(4100) // unreferenced formal parameter
#  include <cuda_fp4.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP4()

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint16_t __cccl_nvfp_get_storage(__half __v) noexcept
{
  struct __helper : __half
  {
    using __half::__x;
  };

  return __helper{__v}.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __half __cccl_make_nvfp16_from_storage(uint16_t __x) noexcept
{
  return __half{__half_raw{__x}};
}

_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_sign_mask     = 0x8000u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_exp_mask      = 0x7c00u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_mant_mask     = 0x03ffu;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_exp_mant_mask = 0x7fffu;
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint16_t __cccl_nvfp_get_storage(__nv_bfloat16 __v) noexcept
{
  struct __helper : __nv_bfloat16
  {
    using __nv_bfloat16::__x;
  };

  return __helper{__v}.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_bfloat16 __cccl_make_nvbf16_from_storage(uint16_t __x) noexcept
{
  return __nv_bfloat16{__nv_bfloat16_raw{__x}};
}

_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_sign_mask     = 0x8000u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_exp_mask      = 0x7f80u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_mant_mask     = 0x007fu;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_exp_mant_mask = 0x7fffu;
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp8_e4m3 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e4m3
__cccl_make_nvfp8_e4m3_from_storage(uint8_t __x) noexcept
{
  __nv_fp8_e4m3 __ret{};
  __ret.__x = static_cast<__nv_fp8_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_sign_mask     = 0x80u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_exp_mask      = 0x78u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_mant_mask     = 0x07u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_exp_mant_mask = 0x7fu;
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp8_e5m2 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e5m2
__cccl_make_nvfp8_e5m2_from_storage(uint8_t __x) noexcept
{
  __nv_fp8_e5m2 __ret{};
  __ret.__x = static_cast<__nv_fp8_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_sign_mask     = 0x80u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_exp_mask      = 0x7cu;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_mant_mask     = 0x03u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_exp_mant_mask = 0x7fu;
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp8_e8m0 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp8_e8m0
__cccl_make_nvfp8_e8m0_from_storage(uint8_t __x) noexcept
{
  __nv_fp8_e8m0 __ret{};
  __ret.__x = static_cast<__nv_fp8_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e8m0_exp_mask = 0xffu;
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp6_e2m3 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e2m3
__cccl_make_nvfp6_e2m3_from_storage(uint8_t __x) noexcept
{
  _CCCL_ASSERT((__x & 0xc0u) == 0u, "Invalid __nv_fp6_e2m3 storage value");
  __nv_fp6_e2m3 __ret{};
  __ret.__x = static_cast<__nv_fp6_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_sign_mask     = 0x20u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_exp_mask      = 0x18u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_mant_mask     = 0x07u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_exp_mant_mask = 0x1fu;
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp6_e3m2 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp6_e3m2
__cccl_make_nvfp6_e3m2_from_storage(uint8_t __x) noexcept
{
  _CCCL_ASSERT((__x & 0xc0u) == 0u, "Invalid __nv_fp6_e3m2 storage value");
  __nv_fp6_e3m2 __ret{};
  __ret.__x = static_cast<__nv_fp6_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_sign_mask     = 0x20u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_exp_mask      = 0x1cu;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_mant_mask     = 0x03u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_exp_mant_mask = 0x1fu;
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_nvfp_get_storage(__nv_fp4_e2m1 __v) noexcept
{
  return __v.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_fp4_e2m1
__cccl_make_nvfp4_e2m1_from_storage(uint8_t __x) noexcept
{
  _CCCL_ASSERT((__x & 0xf0u) == 0u, "Invalid __nv_fp4_e2m1 storage value");
  __nv_fp4_e2m1 __ret{};
  __ret.__x = static_cast<__nv_fp4_storage_t>(__x);
  return __ret;
}

_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_sign_mask     = 0x8u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_exp_mask      = 0x6u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_mant_mask     = 0x1u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_exp_mant_mask = 0x7u;
#endif // _CCCL_HAS_NVFP4_E2M1()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___INTERNAL_NVFP_TYPES_H
