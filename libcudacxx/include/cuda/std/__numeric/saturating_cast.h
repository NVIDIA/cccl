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

// clang-tidy thinks that branches in __saturating_cast_impl_device are all the same.
// NOLINTBEGIN(bugprone-branch-clone)

#if _CCCL_CUDA_COMPILATION()
_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(int8_t)))
[[nodiscard]] _CCCL_DEVICE_API _To __saturating_cast_impl_device(_From __x, int) noexcept
{
  [[maybe_unused]] int __ret;

  if constexpr (sizeof(_From) == sizeof(int8_t))
  {
    if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s8.u8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u8.s8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return static_cast<_To>(__ret);
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int16_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.s8.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s8.u16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u8.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_To>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return static_cast<_To>(__ret);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.s8.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s8.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u8.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_To>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return static_cast<_To>(__ret);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.s8.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s8.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_To>(__ret);
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u8.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_To>(__ret);
    }
    else
    {
      asm("cvt.sat.u8.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return static_cast<_To>(__ret);
    }
  }
  else
  {
    return ::cuda::saturating_overflow_cast<_To>(__x).value;
  }
}

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(int16_t)))
[[nodiscard]] _CCCL_DEVICE_API _To __saturating_cast_impl_device(_From __x, int) noexcept
{
  [[maybe_unused]] _To __ret;

  if constexpr (sizeof(_From) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u16.s8 %0, %1;" : "=h"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int16_t))
  {
    if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s16.u16 %0, %1;" : "=h"(__ret) : "h"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u16.s16 %0, %1;" : "=h"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      // There is a bug on Blackwell this PTX instruction giving invalid result for negative inputs. Enable this once
      // nvbug 6423103 is resolved.
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, ({ return ::cuda::saturating_overflow_cast<_To>(__x).value; }), ({
                          asm("cvt.sat.s16.s32 %0, %1;" : "=h"(__ret) : "r"(__x));
                          return __ret;
                        }))
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s16.u32 %0, %1;" : "=h"(__ret) : "r"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
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
  else if constexpr (sizeof(_From) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      // There is a bug on Blackwell this PTX instruction giving invalid result for negative inputs. Enable this once
      // nvbug 6423103 is resolved.
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100, ({ return ::cuda::saturating_overflow_cast<_To>(__x).value; }), ({
                          asm("cvt.sat.s16.s64 %0, %1;" : "=h"(__ret) : "l"(__x));
                          return __ret;
                        }))
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s16.u64 %0, %1;" : "=h"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
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
    return ::cuda::saturating_overflow_cast<_To>(__x).value;
  }
}

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(int32_t)))
[[nodiscard]] _CCCL_DEVICE_API _To __saturating_cast_impl_device(_From __x, int) noexcept
{
  [[maybe_unused]] _To __ret;

  if constexpr (sizeof(_From) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u32.s8 %0, %1;" : "=r"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int16_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u32.s16 %0, %1;" : "=r"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int32_t))
  {
    if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s32.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u32.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
      return __ret;
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.s32.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s32.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
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
    return ::cuda::saturating_overflow_cast<_To>(__x).value;
  }
}

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(int64_t)))
[[nodiscard]] _CCCL_DEVICE_API _To __saturating_cast_impl_device(_From __x, int) noexcept
{
  [[maybe_unused]] _To __ret;

  if constexpr (sizeof(_From) == sizeof(int8_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u64.s8 %0, %1;" : "=l"(__ret) : "r"(int{__x}));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int16_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u64.s16 %0, %1;" : "=l"(__ret) : "h"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int32_t))
  {
    if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
    {
      asm("cvt.sat.u64.s32 %0, %1;" : "=l"(__ret) : "r"(__x));
      return __ret;
    }
    else
    {
      return static_cast<_To>(__x);
    }
  }
  else if constexpr (sizeof(_From) == sizeof(int64_t))
  {
    if constexpr (is_signed_v<_To> && is_unsigned_v<_From>)
    {
      asm("cvt.sat.s64.u64 %0, %1;" : "=l"(__ret) : "l"(__x));
      return __ret;
    }
    else if constexpr (is_unsigned_v<_To> && is_signed_v<_From>)
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
    return ::cuda::saturating_overflow_cast<_To>(__x).value;
  }
}

template <class _To, class _From>
[[nodiscard]] _CCCL_DEVICE_API _To __saturating_cast_impl_device(_From __x, long) noexcept
{
  return ::cuda::saturating_overflow_cast<_To>(__x).value;
}
#endif // _CCCL_CUDA_COMPILATION()

// NOLINTEND(bugprone-branch-clone)

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES(__cccl_is_integer_v<_To> _CCCL_AND __cccl_is_integer_v<_From>)
[[nodiscard]] _CCCL_API constexpr _To saturating_cast(_From __x) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({ return ::cuda::std::__saturating_cast_impl_device<_To>(__x, 0); }))
  }
  return ::cuda::saturating_overflow_cast<_To>(__x).value;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_SATURATING_CAST_H
