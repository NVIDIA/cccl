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
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp_storage_t<_Tp> __fp_sign_mask_impl() noexcept
{
  constexpr __fp_format __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fmt == __fp_format::__binary16 || __fmt == __fp_format::__bfloat16)
  {
    return __fp_storage_t<_Tp>{0x8000u};
  }
  else if constexpr (__fmt == __fp_format::__binary32)
  {
    return __fp_storage_t<_Tp>{0x80000000u};
  }
  else if constexpr (__fmt == __fp_format::__binary64)
  {
    return __fp_storage_t<_Tp>{0x8000000000000000ull};
  }
  else if constexpr (__fmt == __fp_format::__binary128)
  {
    return __fp_storage_t<_Tp>{0x8000000000000000ull} << 64;
  }
  else if constexpr (__fmt == __fp_format::__fp80_x86)
  {
    return __fp_storage_t<_Tp>{0x8000ull} << 64;
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e4m3 || __fmt == __fp_format::__fp8_nv_e5m2)
  {
    return __fp_storage_t<_Tp>{0x80u};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e8m0)
  {
    return __fp_storage_t<_Tp>{0x00u};
  }
  else if constexpr (__fmt == __fp_format::__fp6_nv_e2m3 || __fmt == __fp_format::__fp6_nv_e3m2)
  {
    return __fp_storage_t<_Tp>{0x20u};
  }
  else if constexpr (__fmt == __fp_format::__fp4_nv_e2m1)
  {
    return __fp_storage_t<_Tp>{0x8u};
  }
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_sign_mask_v = __fp_sign_mask_impl<_Tp>();

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp_storage_t<_Tp> __fp_exp_mask_impl() noexcept
{
  constexpr __fp_format __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fmt == __fp_format::__binary16)
  {
    return __fp_storage_t<_Tp>{0x7c00u};
  }
  else if constexpr (__fmt == __fp_format::__binary32)
  {
    return __fp_storage_t<_Tp>{0x7f800000u};
  }
  else if constexpr (__fmt == __fp_format::__binary64)
  {
    return __fp_storage_t<_Tp>{0x7ff0000000000000ull};
  }
  else if constexpr (__fmt == __fp_format::__binary128)
  {
    return __fp_storage_t<_Tp>{0x7fff000000000000ull} << 64;
  }
  else if constexpr (__fmt == __fp_format::__bfloat16)
  {
    return __fp_storage_t<_Tp>{0x7f80u};
  }
  else if constexpr (__fmt == __fp_format::__fp80_x86)
  {
    return __fp_storage_t<_Tp>{0x7fffull} << 64;
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e4m3)
  {
    return __fp_storage_t<_Tp>{0x78u};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e5m2)
  {
    return __fp_storage_t<_Tp>{0x7cu};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e8m0)
  {
    return __fp_storage_t<_Tp>{0xffu};
  }
  else if constexpr (__fmt == __fp_format::__fp6_nv_e2m3)
  {
    return __fp_storage_t<_Tp>{0x18u};
  }
  else if constexpr (__fmt == __fp_format::__fp6_nv_e3m2)
  {
    return __fp_storage_t<_Tp>{0x1cu};
  }
  else if constexpr (__fmt == __fp_format::__fp4_nv_e2m1)
  {
    return __fp_storage_t<_Tp>{0x6u};
  }
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_exp_mask_v = __fp_exp_mask_impl<_Tp>();

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp_storage_t<_Tp> __fp_mant_mask_impl() noexcept
{
  constexpr __fp_format __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fmt == __fp_format::__binary16)
  {
    return __fp_storage_t<_Tp>{0x03ffu};
  }
  else if constexpr (__fmt == __fp_format::__binary32)
  {
    return __fp_storage_t<_Tp>{0x007fffffu};
  }
  else if constexpr (__fmt == __fp_format::__binary64)
  {
    return __fp_storage_t<_Tp>{0x000fffffffffffffull};
  }
  else if constexpr (__fmt == __fp_format::__binary128)
  {
    return (__fp_storage_t<_Tp>{0x0000ffffffffffffull} << 64) | 0xffffffffffffffffull;
  }
  else if constexpr (__fmt == __fp_format::__bfloat16)
  {
    return __fp_storage_t<_Tp>{0x007fu};
  }
  else if constexpr (__fmt == __fp_format::__fp80_x86)
  {
    return __fp_storage_t<_Tp>{0xffffffffffffffffull};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e4m3)
  {
    return __fp_storage_t<_Tp>{0x07u};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e5m2)
  {
    return __fp_storage_t<_Tp>{0x03u};
  }
  else if constexpr (__fmt == __fp_format::__fp8_nv_e8m0)
  {
    return __fp_storage_t<_Tp>{0x00u};
  }
  else if constexpr (__fmt == __fp_format::__fp6_nv_e2m3)
  {
    return __fp_storage_t<_Tp>{0x07u};
  }
  else if constexpr (__fmt == __fp_format::__fp6_nv_e3m2)
  {
    return __fp_storage_t<_Tp>{0x03u};
  }
  else if constexpr (__fmt == __fp_format::__fp4_nv_e2m1)
  {
    return __fp_storage_t<_Tp>{0x01u};
  }
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported floating point format");
  }
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_mant_mask_v = __fp_mant_mask_impl<_Tp>();

template <class _Tp>
_CCCL_INLINE_VAR constexpr __fp_storage_t<_Tp> __fp_exp_mant_mask_v =
  static_cast<__fp_storage_t<_Tp>>(__fp_exp_mask_v<_Tp> | __fp_mant_mask_v<_Tp>);

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_MASK_H
