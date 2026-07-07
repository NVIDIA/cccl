//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_SATURATING_CAST_H
#define _CUDA_STD___NUMERIC_SATURATING_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/saturating_overflow_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CUDA_COMPILATION()
_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES((sizeof(_Tp) == sizeof(int8_t)))
[[nodiscard]] _CCCL_DEVICE_API _Up __saturating_cast_impl_device(_Tp __x, int) noexcept
{
  [[maybe_unused]] int __ret;

  if constexpr (sizeof(_Tp) == sizeof(int8_t))
  {
    if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s8.u8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u8.s8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return static_cast<_Up>(__ret);
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int16_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.s8.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s8.u16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u8.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_Up>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_Up>(__ret);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.s8.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s8.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u8.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_Up>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_Up>(__ret);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.s8.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s8.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_Up>(__ret);
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u8.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_Up>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_Up>(__ret);
    }
  }
  else
  {
    return ::cuda::saturating_overflow_cast<_Up>(__x).value;
  }
}

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES((sizeof(_Tp) == sizeof(int16_t)))
[[nodiscard]] _CCCL_DEVICE_API _Up __saturating_cast_impl_device(_Tp __x, int) noexcept
{
  [[maybe_unused]] _Up __ret;

  if constexpr (sizeof(_Tp) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u16.s8 %0, %1;" : "=h"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int16_t))
  {
    if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s16.u16 %0, %1;" : "=h"(__ret) : "h"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u16.s16 %0, %1;" : "=h"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      // There is a bug on Blackwell this PTX instruction giving invalid result for negative inputs. Enable this once
      // nvbug 6423103 is resolved.
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, ({ return ::cuda::saturating_overflow_cast<_Up>(__x).value; }), ({
                          asm("cvt.sat.s16.s32 %0, %1;" : "=h"(__ret) : "r"(__x));
                          return __ret;
                        }))
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s16.u32 %0, %1;" : "=h"(__ret) : "r"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u16.s32 %0, %1;" : "=h"(__ret) : "r"(__x));
      return __ret;
    }
    else
    {
      asm("cvt.sat.u16.u32 %0, %1;" : "=h"(__ret) : "r"(__x));
      return __ret;
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      // There is a bug on Blackwell this PTX instruction giving invalid result for negative inputs. Enable this once
      // nvbug 6423103 is resolved.
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, ({ return ::cuda::saturating_overflow_cast<_Up>(__x).value; }), ({
                          asm("cvt.sat.s16.s64 %0, %1;" : "=h"(__ret) : "l"(__x));
                          return __ret;
                        }))
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s16.u64 %0, %1;" : "=h"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u16.s64 %0, %1;" : "=h"(__ret) : "l"(__x));
      return __ret;
    }
    else
    {
      asm("cvt.sat.u16.u64 %0, %1;" : "=h"(__ret) : "l"(__x));
      return __ret;
    }
  }
  else
  {
    return ::cuda::saturating_overflow_cast<_Up>(__x).value;
  }
}

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES((sizeof(_Tp) == sizeof(int32_t)))
[[nodiscard]] _CCCL_DEVICE_API _Up __saturating_cast_impl_device(_Tp __x, int) noexcept
{
  [[maybe_unused]] _Up __ret;

  if constexpr (sizeof(_Tp) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u32.s8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int16_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u32.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s32.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u32.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return __ret;
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.s32.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s32.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u32.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
    else
    {
      asm("cvt.sat.u32.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
  }
  else
  {
    return ::cuda::saturating_overflow_cast<_Up>(__x).value;
  }
}

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES((sizeof(_Tp) == sizeof(int64_t)))
[[nodiscard]] _CCCL_DEVICE_API _Up __saturating_cast_impl_device(_Tp __x, int) noexcept
{
  [[maybe_unused]] _Up __ret;

  if constexpr (sizeof(_Tp) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u64.s8 %0, %1;" : "=l"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int16_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u64.s16 %0, %1;" : "=l"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int32_t))
  {
    if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u64.s32 %0, %1;" : "=l"(__ret) : "r"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_Up>(__x);
    }
  }
  else if constexpr (sizeof(_Tp) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_Up> && is_unsigned_v<_Tp>)
    {
      asm("cvt.sat.s64.u64 %0, %1;" : "=l"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_Up> && is_signed_v<_Tp>)
    {
      asm("cvt.sat.u64.s64 %0, %1;" : "=l"(__ret) : "l"(__x));
      return __ret;
    }
    else
    {
      return __x;
    }
  }
  else
  {
    return ::cuda::saturating_overflow_cast<_Up>(__x).value;
  }
}

template <class _Up, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Up __saturating_cast_impl_device(_Tp __x, long) noexcept
{
  return ::cuda::saturating_overflow_cast<_Up>(__x).value;
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Up> _CCCL_AND __cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Up saturating_cast(_Tp __x) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({ return ::cuda::std::__saturating_cast_impl_device<_Up>(__x, 0); }))
  }
  return ::cuda::saturating_overflow_cast<_Up>(__x).value;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_SATURATING_CAST_H
