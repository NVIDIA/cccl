//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_MASK_H
#define _LIBCUDACXX___FLOATING_POINT_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_sign_mask_impl() noexcept
{
  if constexpr (_CCCL_TRAIT(is_same, _Tp, float))
  {
    return __fp_storage_t<float>{0x80000000u};
  }
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, double))
  {
    return __fp_storage_t<double>{0x8000000000000000ull};
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    return __fp_storage_t<__half>{0x8000u};
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    return __fp_storage_t<__nv_bfloat16>{0x8000u};
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    return __fp_storage_t<__nv_fp8_e4m3>{0x80u};
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    return __fp_storage_t<__nv_fp8_e5m2>{0x80u};
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    return __fp_storage_t<__nv_fp8_e8m0>{0x00u};
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    return __fp_storage_t<__nv_fp6_e2m3>{0x20u};
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    return __fp_storage_t<__nv_fp6_e3m2>{0x20u};
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    return __fp_storage_t<__nv_fp4_e2m1>{0x8u};
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating-point type");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_sign_mask_v = __fp_sign_mask_impl<_Tp>();

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_exp_mask_impl() noexcept
{
  if constexpr (_CCCL_TRAIT(is_same, _Tp, float))
  {
    return __fp_storage_t<float>{0x7f800000u};
  }
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, double))
  {
    return __fp_storage_t<double>{0x7ff0000000000000ull};
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    return __fp_storage_t<__half>{0x7c00u};
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    return __fp_storage_t<__nv_bfloat16>{0x7f80u};
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    return __fp_storage_t<__nv_fp8_e4m3>{0x78u};
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    return __fp_storage_t<__nv_fp8_e5m2>{0x7cu};
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    return __fp_storage_t<__nv_fp8_e8m0>{0xffu};
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    return __fp_storage_t<__nv_fp6_e2m3>{0x18u};
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    return __fp_storage_t<__nv_fp6_e3m2>{0x1cu};
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    return __fp_storage_t<__nv_fp4_e2m1>{0x6u};
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating-point type");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_exp_mask_v = __fp_exp_mask_impl<_Tp>();

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_mant_mask_impl() noexcept
{
  if constexpr (_CCCL_TRAIT(is_same, _Tp, float))
  {
    return __fp_storage_t<float>{0x007fffffu};
  }
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, double))
  {
    return __fp_storage_t<double>{0x000fffffffffffffull};
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    return __fp_storage_t<__half>{0x03ffu};
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    return __fp_storage_t<__nv_bfloat16>{0x007fu};
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    return __fp_storage_t<__nv_fp8_e4m3>{0x07u};
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    return __fp_storage_t<__nv_fp8_e5m2>{0x03u};
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    return __fp_storage_t<__nv_fp8_e8m0>{0x00u};
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    return __fp_storage_t<__nv_fp6_e2m3>{0x07u};
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    return __fp_storage_t<__nv_fp6_e3m2>{0x03u};
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    return __fp_storage_t<__nv_fp4_e2m1>{0x01u};
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating-point type");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_mant_mask_v = __fp_mant_mask_impl<_Tp>();

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_exp_mant_mask_v =
  static_cast<__fp_storage_t<_Tp>>(__fp_exp_mask_v<_Tp> | __fp_mant_mask_v<_Tp>);

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_MASK_H
