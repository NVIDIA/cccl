//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FP_UTILS_H
#define _LIBCUDACXX___CMATH_FP_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__internal/nvfp_types.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fp32

_CCCL_INLINE_VAR constexpr uint32_t __cccl_fp32_sign_mask     = 0x80000000u;
_CCCL_INLINE_VAR constexpr uint32_t __cccl_fp32_exp_mask      = 0x7f800000u;
_CCCL_INLINE_VAR constexpr uint32_t __cccl_fp32_mant_mask     = 0x007fffffu;
_CCCL_INLINE_VAR constexpr uint32_t __cccl_fp32_exp_mant_mask = 0x7fffffffu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST uint32_t
__cccl_fp_get_storage(float __v) noexcept
{
  return _CUDA_VSTD::bit_cast<uint32_t>(__v);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST float
__cccl_make_fp32_from_storage(uint32_t __x) noexcept
{
  return _CUDA_VSTD::bit_cast<float>(__x);
}

// fp64

_CCCL_INLINE_VAR constexpr uint64_t __cccl_fp64_sign_mask     = 0x8000000000000000ull;
_CCCL_INLINE_VAR constexpr uint64_t __cccl_fp64_exp_mask      = 0x7ff0000000000000ull;
_CCCL_INLINE_VAR constexpr uint64_t __cccl_fp64_mant_mask     = 0x000fffffffffffffull;
_CCCL_INLINE_VAR constexpr uint64_t __cccl_fp64_exp_mant_mask = 0x7fffffffffffffffull;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST uint64_t
__cccl_fp_get_storage(double __v) noexcept
{
  return _CUDA_VSTD::bit_cast<uint64_t>(__v);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST double
__cccl_make_fp64_from_storage(uint64_t __x) noexcept
{
  return _CUDA_VSTD::bit_cast<double>(__x);
}

// nvfp16

#if _CCCL_HAS_NVFP16()
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_sign_mask     = 0x8000u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_exp_mask      = 0x7c00u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_mant_mask     = 0x03ffu;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvfp16_exp_mant_mask = 0x7fffu;

struct __cccl_nvfp16_manip_helper : __half
{
  using __half::__x;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint16_t __cccl_fp_get_storage(__half __v) noexcept
{
  return __cccl_nvfp16_manip_helper{__v}.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __half __cccl_make_nvfp16_from_storage(uint16_t __x) noexcept
{
  __cccl_nvfp16_manip_helper __helper{};
  __helper.__x = __x;
  return __helper;
}
#endif // _CCCL_HAS_NVFP16()

// nvbf16

#if _CCCL_HAS_NVBF16()
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_sign_mask     = 0x8000u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_exp_mask      = 0x7f80u;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_mant_mask     = 0x007fu;
_CCCL_INLINE_VAR constexpr uint16_t __cccl_nvbf16_exp_mant_mask = 0x7fffu;

struct __cccl_nvbf16_manip_helper : __nv_bfloat16
{
  using __nv_bfloat16::__x;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint16_t __cccl_fp_get_storage(__nv_bfloat16 __v) noexcept
{
  return __cccl_nvbf16_manip_helper{__v}.__x;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __nv_bfloat16 __cccl_make_nvbf16_from_storage(uint16_t __x) noexcept
{
  __cccl_nvbf16_manip_helper __helper{};
  __helper.__x = __x;
  return __helper;
}
#endif // _CCCL_HAS_NVBF16()

// nvfp8_e4m3

#if _CCCL_HAS_NVFP8_E4M3()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_sign_mask     = 0x80u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_exp_mask      = 0x78u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_mant_mask     = 0x07u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e4m3_exp_mant_mask = 0x7fu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp8_e4m3 __v) noexcept
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
#endif // _CCCL_HAS_NVFP8_E4M3()

// nvfp8_e5m2

#if _CCCL_HAS_NVFP8_E5M2()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_sign_mask     = 0x80u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_exp_mask      = 0x7cu;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_mant_mask     = 0x03u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e5m2_exp_mant_mask = 0x7fu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp8_e5m2 __v) noexcept
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
#endif // _CCCL_HAS_NVFP8_E5M2()

// nvfp8_e8m0

#if _CCCL_HAS_NVFP8_E8M0()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp8_e8m0_exp_mask = 0xffu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp8_e8m0 __v) noexcept
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
#endif // _CCCL_HAS_NVFP8_E8M0()

// nvfp6_e2m3

#if _CCCL_HAS_NVFP6_E2M3()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_sign_mask     = 0x20u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_exp_mask      = 0x18u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_mant_mask     = 0x07u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e2m3_exp_mant_mask = 0x1fu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp6_e2m3 __v) noexcept
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
#endif // _CCCL_HAS_NVFP6_E2M3()

// nvfp6_e3m2

#if _CCCL_HAS_NVFP6_E3M2()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_sign_mask     = 0x20u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_exp_mask      = 0x1cu;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_mant_mask     = 0x03u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp6_e3m2_exp_mant_mask = 0x1fu;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp6_e3m2 __v) noexcept
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
#endif // _CCCL_HAS_NVFP6_E3M2()

// nvfp4_e2m1

#if _CCCL_HAS_NVFP4_E2M1()
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_sign_mask     = 0x8u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_exp_mask      = 0x6u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_mant_mask     = 0x1u;
_CCCL_INLINE_VAR constexpr uint8_t __cccl_nvfp4_e2m1_exp_mant_mask = 0x7u;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr uint8_t __cccl_fp_get_storage(__nv_fp4_e2m1 __v) noexcept
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
#endif // _CCCL_HAS_NVFP4_E2M1()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_FP_UTILS_H
